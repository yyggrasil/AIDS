"""
Flow Aggregator Module for Network Intrusion Detection.
Captures network packets, aggregates them into bidirectional flows (5-tuple),
and extracts 70 statistical features matching the CICFlowMeter standard for
the trained Stacking Classifier pipeline.
"""

import time
import math
import logging
from collections import defaultdict
import numpy as np
import pandas as pd

logger = logging.getLogger("AIDS.FlowAggregator")

# Standard 70 features expected by the trained pipeline
FEATURE_NAMES = [
    'Src Port', 'Dst Port', 'Protocol', 'Flow Duration', 'Total Fwd Packet',
    'Total Bwd packets', 'Total Length of Fwd Packet', 'Total Length of Bwd Packet',
    'Fwd Packet Length Max', 'Fwd Packet Length Min', 'Fwd Packet Length Mean',
    'Fwd Packet Length Std', 'Bwd Packet Length Max', 'Bwd Packet Length Min',
    'Bwd Packet Length Mean', 'Bwd Packet Length Std', 'Flow Bytes/s',
    'Flow Packets/s', 'Flow IAT Mean', 'Flow IAT Std', 'Flow IAT Max',
    'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean', 'Fwd IAT Std',
    'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
    'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags',
    'Fwd Header Length', 'Bwd Header Length', 'Fwd Packets/s', 'Bwd Packets/s',
    'Packet Length Min', 'Packet Length Max', 'Packet Length Mean',
    'Packet Length Std', 'Packet Length Variance', 'FIN Flag Count',
    'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count', 'ACK Flag Count',
    'Down/Up Ratio', 'Average Packet Size', 'Fwd Segment Size Avg',
    'Bwd Segment Size Avg', 'Bwd Bytes/Bulk Avg', 'Bwd Packet/Bulk Avg',
    'Bwd Bulk Rate Avg', 'Subflow Fwd Packets', 'Subflow Fwd Bytes',
    'Subflow Bwd Packets', 'Subflow Bwd Bytes', 'FWD Init Win Bytes',
    'Bwd Init Win Bytes', 'Fwd Act Data Pkts', 'Fwd Seg Size Min',
    'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
]


def _calc_stats(values):
    """Returns (min, max, mean, std, var) for a list of values."""
    if not values:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    arr = np.array(values, dtype=np.float64)
    v_min = float(np.min(arr))
    v_max = float(np.max(arr))
    v_mean = float(np.mean(arr))
    if len(arr) > 1:
        v_std = float(np.std(arr, ddof=1))
        v_var = float(np.var(arr, ddof=1))
    else:
        v_std = 0.0
        v_var = 0.0
    return v_min, v_max, v_mean, v_std, v_var


def _calc_iats(timestamps):
    """Calculates Inter-Arrival Times in microseconds from a list of second timestamps."""
    if len(timestamps) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    diffs = [(timestamps[i] - timestamps[i - 1]) * 1e6 for i in range(1, len(timestamps))]
    total = float(sum(diffs))
    arr = np.array(diffs, dtype=np.float64)
    iat_min = float(np.min(arr))
    iat_max = float(np.max(arr))
    iat_mean = float(np.mean(arr))
    iat_std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return total, iat_mean, iat_std, iat_max, iat_min


class Flow:
    """
    Represents a bidirectional network flow identified by 5-tuple:
    (src_ip, src_port, dst_ip, dst_port, protocol).
    """

    def __init__(self, src_ip, src_port, dst_ip, dst_port, protocol, start_time, idle_threshold=5.0):
        self.src_ip = src_ip
        self.src_port = int(src_port)
        self.dst_ip = dst_ip
        self.dst_port = int(dst_port)
        self.protocol = str(protocol)
        
        self.start_time = float(start_time)
        self.last_time = float(start_time)
        self.idle_threshold = float(idle_threshold)  # in seconds

        # Packet length trackers
        self.fwd_packet_lengths = []
        self.bwd_packet_lengths = []
        self.all_packet_lengths = []

        # Timestamps
        self.fwd_timestamps = []
        self.bwd_timestamps = []
        self.all_timestamps = []

        # Header lengths (bytes)
        self.fwd_header_lengths = []
        self.bwd_header_lengths = []

        # TCP Flag Counters
        self.fwd_psh_flags = 0
        self.fin_flags = 0
        self.syn_flags = 0
        self.rst_flags = 0
        self.psh_flags = 0
        self.ack_flags = 0

        # TCP Window & Segment attributes
        self.fwd_init_win = 0
        self.bwd_init_win = 0
        self.fwd_act_data_pkts = 0
        self.fwd_seg_size_min = 20  # Default minimum TCP header size

        # Active / Idle time measurement
        self.active_times = []
        self.idle_times = []
        self.current_active_start = float(start_time)
        self.current_active_end = float(start_time)

        # State flags
        self.is_finished = False

    def add_packet(self, is_fwd: bool, pkt_len: int, header_len: int, timestamp: float,
                   tcp_flags: dict = None, tcp_window: int = 0, payload_len: int = 0):
        """Adds a packet to the flow and updates flow statistics."""
        timestamp = float(timestamp)
        pkt_len = int(pkt_len)
        header_len = int(header_len)
        payload_len = int(payload_len)

        # Check Active / Idle interval
        if len(self.all_timestamps) > 0:
            gap = timestamp - self.last_time
            if gap > self.idle_threshold:
                # Gap exceeded idle threshold: record completed active and idle periods
                active_dur = (self.current_active_end - self.current_active_start) * 1e6
                if active_dur > 0:
                    self.active_times.append(active_dur)
                self.idle_times.append(gap * 1e6)
                self.current_active_start = timestamp
                self.current_active_end = timestamp
            else:
                self.current_active_end = timestamp
        else:
            self.current_active_start = timestamp
            self.current_active_end = timestamp

        self.last_time = timestamp
        self.all_timestamps.append(timestamp)
        self.all_packet_lengths.append(pkt_len)

        if is_fwd:
            self.fwd_timestamps.append(timestamp)
            self.fwd_packet_lengths.append(pkt_len)
            self.fwd_header_lengths.append(header_len)
            if header_len > 0 and (header_len < self.fwd_seg_size_min or len(self.fwd_header_lengths) == 1):
                self.fwd_seg_size_min = header_len
            if len(self.fwd_timestamps) == 1 and tcp_window > 0:
                self.fwd_init_win = int(tcp_window)
            if payload_len > 0:
                self.fwd_act_data_pkts += 1
            if tcp_flags and tcp_flags.get('PSH', False):
                self.fwd_psh_flags += 1
        else:
            self.bwd_timestamps.append(timestamp)
            self.bwd_packet_lengths.append(pkt_len)
            self.bwd_header_lengths.append(header_len)
            if len(self.bwd_timestamps) == 1 and tcp_window > 0:
                self.bwd_init_win = int(tcp_window)

        # Global TCP flags count
        if tcp_flags:
            if tcp_flags.get('FIN', False):
                self.fin_flags += 1
                self.is_finished = True
            if tcp_flags.get('SYN', False):
                self.syn_flags += 1
            if tcp_flags.get('RST', False):
                self.rst_flags += 1
                self.is_finished = True
            if tcp_flags.get('PSH', False):
                self.psh_flags += 1
            if tcp_flags.get('ACK', False):
                self.ack_flags += 1

    def is_expired(self, current_time: float, inactivity_timeout: float = 15.0, active_timeout: float = 120.0) -> bool:
        """Determines if the flow should be terminated and classified."""
        if self.is_finished:
            return True
        now = float(current_time)
        if (now - self.last_time) >= inactivity_timeout:
            return True
        if (now - self.start_time) >= active_timeout:
            return True
        return False

    def extract_features(self) -> dict:
        """Extracts the exact 70 features needed by the preprocessor and Stacking model."""
        duration_sec = max(0.0, self.last_time - self.start_time)
        duration_us = duration_sec * 1e6
        # Single packet fallback
        if duration_us <= 0.0:
            duration_us = 1.0
            duration_sec = 1e-6

        total_fwd_pkts = len(self.fwd_packet_lengths)
        total_bwd_pkts = len(self.bwd_packet_lengths)
        total_pkts = total_fwd_pkts + total_bwd_pkts

        tot_len_fwd = sum(self.fwd_packet_lengths)
        tot_len_bwd = sum(self.bwd_packet_lengths)
        tot_bytes = tot_len_fwd + tot_len_bwd

        # Packet Length Statistics
        f_min, f_max, f_mean, f_std, _ = _calc_stats(self.fwd_packet_lengths)
        b_min, b_max, b_mean, b_std, _ = _calc_stats(self.bwd_packet_lengths)
        all_min, all_max, all_mean, all_std, all_var = _calc_stats(self.all_packet_lengths)

        # Rates
        flow_bytes_s = tot_bytes / duration_sec if duration_sec > 0 else 0.0
        flow_pkts_s = total_pkts / duration_sec if duration_sec > 0 else 0.0
        fwd_pkts_s = total_fwd_pkts / duration_sec if duration_sec > 0 else 0.0
        bwd_pkts_s = total_bwd_pkts / duration_sec if duration_sec > 0 else 0.0

        # Inter-Arrival Times
        _, flow_iat_mean, flow_iat_std, flow_iat_max, flow_iat_min = _calc_iats(self.all_timestamps)
        fwd_iat_tot, fwd_iat_mean, fwd_iat_std, fwd_iat_max, fwd_iat_min = _calc_iats(self.fwd_timestamps)
        bwd_iat_tot, bwd_iat_mean, bwd_iat_std, bwd_iat_max, bwd_iat_min = _calc_iats(self.bwd_timestamps)

        # Header lengths sum
        fwd_header_len = sum(self.fwd_header_lengths)
        bwd_header_len = sum(self.bwd_header_lengths)

        # Down / Up Ratio
        down_up_ratio = (total_bwd_pkts / total_fwd_pkts) if total_fwd_pkts > 0 else 0.0

        # Average packet size and segment sizes
        avg_pkt_size = (tot_bytes / total_pkts) if total_pkts > 0 else 0.0
        fwd_seg_size_avg = f_mean
        bwd_seg_size_avg = b_mean

        # Active / Idle Times
        final_active_dur = (self.current_active_end - self.current_active_start) * 1e6
        active_list = list(self.active_times)
        if final_active_dur > 0 or not active_list:
            active_list.append(max(final_active_dur, 0.0))

        idle_list = list(self.idle_times)

        act_min, act_max, act_mean, act_std, _ = _calc_stats(active_list)
        idle_min, idle_max, idle_mean, idle_std, _ = _calc_stats(idle_list)

        features = {
            'Src Port': float(self.src_port),
            'Dst Port': float(self.dst_port),
            'Protocol': str(self.protocol),
            'Flow Duration': float(duration_us),
            'Total Fwd Packet': float(total_fwd_pkts),
            'Total Bwd packets': float(total_bwd_pkts),
            'Total Length of Fwd Packet': float(tot_len_fwd),
            'Total Length of Bwd Packet': float(tot_len_bwd),
            'Fwd Packet Length Max': float(f_max),
            'Fwd Packet Length Min': float(f_min),
            'Fwd Packet Length Mean': float(f_mean),
            'Fwd Packet Length Std': float(f_std),
            'Bwd Packet Length Max': float(b_max),
            'Bwd Packet Length Min': float(b_min),
            'Bwd Packet Length Mean': float(b_mean),
            'Bwd Packet Length Std': float(b_std),
            'Flow Bytes/s': float(flow_bytes_s),
            'Flow Packets/s': float(flow_pkts_s),
            'Flow IAT Mean': float(flow_iat_mean),
            'Flow IAT Std': float(flow_iat_std),
            'Flow IAT Max': float(flow_iat_max),
            'Flow IAT Min': float(flow_iat_min),
            'Fwd IAT Total': float(fwd_iat_tot),
            'Fwd IAT Mean': float(fwd_iat_mean),
            'Fwd IAT Std': float(fwd_iat_std),
            'Fwd IAT Max': float(fwd_iat_max),
            'Fwd IAT Min': float(fwd_iat_min),
            'Bwd IAT Total': float(bwd_iat_tot),
            'Bwd IAT Mean': float(bwd_iat_mean),
            'Bwd IAT Std': float(bwd_iat_std),
            'Bwd IAT Max': float(bwd_iat_max),
            'Bwd IAT Min': float(bwd_iat_min),
            'Fwd PSH Flags': float(self.fwd_psh_flags),
            'Fwd Header Length': float(fwd_header_len),
            'Bwd Header Length': float(bwd_header_len),
            'Fwd Packets/s': float(fwd_pkts_s),
            'Bwd Packets/s': float(bwd_pkts_s),
            'Packet Length Min': float(all_min),
            'Packet Length Max': float(all_max),
            'Packet Length Mean': float(all_mean),
            'Packet Length Std': float(all_std),
            'Packet Length Variance': float(all_var),
            'FIN Flag Count': float(self.fin_flags),
            'SYN Flag Count': float(self.syn_flags),
            'RST Flag Count': float(self.rst_flags),
            'PSH Flag Count': float(self.psh_flags),
            'ACK Flag Count': float(self.ack_flags),
            'Down/Up Ratio': float(down_up_ratio),
            'Average Packet Size': float(avg_pkt_size),
            'Fwd Segment Size Avg': float(fwd_seg_size_avg),
            'Bwd Segment Size Avg': float(bwd_seg_size_avg),
            'Bwd Bytes/Bulk Avg': 0.0,
            'Bwd Packet/Bulk Avg': 0.0,
            'Bwd Bulk Rate Avg': 0.0,
            'Subflow Fwd Packets': float(total_fwd_pkts),
            'Subflow Fwd Bytes': float(tot_len_fwd),
            'Subflow Bwd Packets': float(total_bwd_pkts),
            'Subflow Bwd Bytes': float(tot_len_bwd),
            'FWD Init Win Bytes': float(self.fwd_init_win),
            'Bwd Init Win Bytes': float(self.bwd_init_win),
            'Fwd Act Data Pkts': float(self.fwd_act_data_pkts),
            'Fwd Seg Size Min': float(self.fwd_seg_size_min),
            'Active Mean': float(act_mean),
            'Active Std': float(act_std),
            'Active Max': float(act_max),
            'Active Min': float(act_min),
            'Idle Mean': float(idle_mean),
            'Idle Std': float(idle_std),
            'Idle Max': float(idle_max),
            'Idle Min': float(idle_min)
        }
        return features

    def to_dataframe(self) -> pd.DataFrame:
        """Converts the extracted features into a single-row Pandas DataFrame."""
        return pd.DataFrame([self.extract_features()])

    def get_summary(self) -> dict:
        """Returns a high-level summary of the flow for logging and email alerts."""
        duration = max(0.0, self.last_time - self.start_time)
        proto_name = "TCP" if self.protocol == '6' else ("UDP" if self.protocol == '17' else f"Proto-{self.protocol}")
        return {
            'src_ip': self.src_ip,
            'src_port': self.src_port,
            'dst_ip': self.dst_ip,
            'dst_port': self.dst_port,
            'protocol': self.protocol,
            'protocol_name': proto_name,
            'start_time': self.start_time,
            'last_time': self.last_time,
            'duration_sec': round(duration, 4),
            'total_packets': len(self.all_packet_lengths),
            'total_bytes': sum(self.all_packet_lengths),
            'fwd_packets': len(self.fwd_packet_lengths),
            'bwd_packets': len(self.bwd_packet_lengths)
        }


class FlowAggregator:
    """
    Manages active network flows in real-time, aggregating incoming packets
    and flushing expired flows for classification.
    Optimized for low RAM consumption on Raspberry Pi.
    """

    def __init__(self, inactivity_timeout: float = 15.0, active_timeout: float = 120.0,
                 max_flows: int = 10000, idle_threshold: float = 5.0):
        self.inactivity_timeout = float(inactivity_timeout)
        self.active_timeout = float(active_timeout)
        self.max_flows = int(max_flows)
        self.idle_threshold = float(idle_threshold)
        
        # Active flows map: fwd_key -> Flow
        self.flows = {}

        self.total_packets_processed = 0
        self.total_flows_flushed = 0

    def _get_packet_info(self, packet):
        """
        Parses Scapy packet and extracts 5-tuple, lengths, flags, and headers.
        Returns None if packet is not IPv4 TCP/UDP.
        """
        try:
            from scapy.layers.inet import IP, TCP, UDP
        except ImportError:
            return None

        # Check for IP layer (supports Ethernet, SLL, Loopback, Raw IP)
        if not (hasattr(packet, 'haslayer') and packet.haslayer(IP)):
            try:
                packet = IP(bytes(packet))
            except Exception:
                return None
            if not (hasattr(packet, 'haslayer') and packet.haslayer(IP)):
                return None

        ip_layer = packet[IP]
        src_ip = ip_layer.src
        dst_ip = ip_layer.dst
        proto_num = ip_layer.proto
        pkt_len = len(packet)
        timestamp = float(packet.time if hasattr(packet, 'time') and packet.time else time.time())

        src_port = 0
        dst_port = 0
        header_len = 20  # Default IP header
        tcp_window = 0
        payload_len = 0
        tcp_flags = None

        if proto_num == 6 and packet.haslayer(TCP):  # TCP
            tcp_layer = packet[TCP]
            src_port = int(tcp_layer.sport)
            dst_port = int(tcp_layer.dport)
            header_len = int(tcp_layer.dataofs * 4) if hasattr(tcp_layer, 'dataofs') and tcp_layer.dataofs else 20
            tcp_window = int(tcp_layer.window) if hasattr(tcp_layer, 'window') else 0
            payload_len = len(tcp_layer.payload) if hasattr(tcp_layer, 'payload') else 0
            
            flags_str = str(tcp_layer.flags)
            tcp_flags = {
                'FIN': 'F' in flags_str,
                'SYN': 'S' in flags_str,
                'RST': 'R' in flags_str,
                'PSH': 'P' in flags_str,
                'ACK': 'A' in flags_str,
                'URG': 'U' in flags_str,
                'ECE': 'E' in flags_str,
                'CWR': 'C' in flags_str
            }
        elif proto_num == 17 and packet.haslayer(UDP):  # UDP
            udp_layer = packet[UDP]
            src_port = int(udp_layer.sport)
            dst_port = int(udp_layer.dport)
            header_len = 8
            payload_len = len(udp_layer.payload) if hasattr(udp_layer, 'payload') else 0
        else:
            # Non-TCP/UDP IP packet
            return None

        return {
            'src_ip': src_ip,
            'src_port': src_port,
            'dst_ip': dst_ip,
            'dst_port': dst_port,
            'proto': str(proto_num),
            'pkt_len': pkt_len,
            'header_len': header_len,
            'timestamp': timestamp,
            'tcp_flags': tcp_flags,
            'tcp_window': tcp_window,
            'payload_len': payload_len
        }

    def process_packet(self, packet):
        """Processes a single incoming packet and updates the corresponding flow."""
        info = self._get_packet_info(packet)
        if not info:
            return None

        self.total_packets_processed += 1
        return self.add_packet_info(info)

    def add_packet_info(self, info: dict):
        """Adds parsed packet information to the flow state."""
        src_ip = info['src_ip']
        src_port = info['src_port']
        dst_ip = info['dst_ip']
        dst_port = info['dst_port']
        proto = info['proto']
        timestamp = info['timestamp']

        fwd_key = (src_ip, src_port, dst_ip, dst_port, proto)
        bwd_key = (dst_ip, dst_port, src_ip, src_port, proto)

        if fwd_key in self.flows:
            flow = self.flows[fwd_key]
            is_fwd = True
        elif bwd_key in self.flows:
            flow = self.flows[bwd_key]
            is_fwd = False
        else:
            # New flow: check capacity limit for Raspberry Pi
            if len(self.flows) >= self.max_flows:
                self._evict_oldest_flows(count=max(10, int(self.max_flows * 0.05)))

            flow = Flow(
                src_ip=src_ip,
                src_port=src_port,
                dst_ip=dst_ip,
                dst_port=dst_port,
                protocol=proto,
                start_time=timestamp,
                idle_threshold=self.idle_threshold
            )
            self.flows[fwd_key] = flow
            is_fwd = True

        flow.add_packet(
            is_fwd=is_fwd,
            pkt_len=info['pkt_len'],
            header_len=info['header_len'],
            timestamp=timestamp,
            tcp_flags=info.get('tcp_flags'),
            tcp_window=info.get('tcp_window', 0),
            payload_len=info.get('payload_len', 0)
        )

        return flow

    def flush_expired(self, current_time: float = None, force: bool = False) -> list:
        """
        Scans all active flows, removes expired ones from memory, and returns them
        for inference/classification. If force=True, flushes all flows immediately.
        """
        now = float(current_time if current_time is not None else time.time())
        expired_flows = []
        keys_to_remove = []

        for key, flow in self.flows.items():
            if force or flow.is_expired(now, inactivity_timeout=self.inactivity_timeout, active_timeout=self.active_timeout):
                expired_flows.append(flow)
                keys_to_remove.append(key)

        for key in keys_to_remove:
            del self.flows[key]

        self.total_flows_flushed += len(expired_flows)
        return expired_flows


    def _evict_oldest_flows(self, count: int = 100):
        """Evicts oldest flows when max capacity is reached to prevent OOM on Raspberry Pi."""
        if not self.flows:
            return
        sorted_flows = sorted(self.flows.items(), key=lambda item: item[1].last_time)
        to_evict = sorted_flows[:count]
        for key, _ in to_evict:
            del self.flows[key]
        logger.warning("Memória restrita: %d fluxos antigos foram descartados para liberar RAM.", len(to_evict))

    def get_stats(self) -> dict:
        """Returns aggregator operational statistics."""
        return {
            'active_flows': len(self.flows),
            'total_packets_processed': self.total_packets_processed,
            'total_flows_flushed': self.total_flows_flushed
        }
