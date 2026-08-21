# Diretrizes de Contribuição

Obrigado pelo interesse em contribuir para o projeto **AIDS - Alternative Traffic Detection**!

## Como Contribuir

1. **Fork do Repositório**:
   Crie uma cópia do projeto em seu GitHub.

2. **Clone e Criação de Branch**:
   ```bash
   git clone https://github.com/yyggrasil/AIDS_alternativo.git
   cd AIDS_alternativo
   git checkout -b feature/minha-melhoria
   ```

3. **Configuração do Ambiente**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # No Windows: .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

4. **Padrões de Código**:
   - Mantenha a tipagem e comentários claros em funções de Machine Learning.
   - Siga as convenções da PEP 8.
   - Certifique-se de que novos modelos ou transformações respeitem a modularidade em `src/`.

5. **Testes**:
   Execute os testes e o pipeline reduzido antes de submeter:
   ```bash
   python -m unittest discover tests/
   ```

6. **Submissão do Pull Request (PR)**:
   - Escreva mensagens de commit convencionais (`feat:`, `fix:`, `docs:`, `refactor:`).
   - Abra um Pull Request detalhando as alterações e os resultados obtidos.
