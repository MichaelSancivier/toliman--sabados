# 🧭 Conferência de Configuração por Placa

Aplicação **Streamlit** para verificar se a frota de um cliente está configurada conforme as regras desejadas.  
Mostra apenas as **placas divergentes** da configuração-alvo.

## 🚀 Como usar

1. **Clonar o repositório** e instalar dependências:
   ```bash
   pip install -r requirements.txt
   ```

2. **Rodar o app**:
   ```bash
   streamlit run streamlit_app.py
   ```

3. **No navegador**:
   - Faça upload de um ou mais CSVs de base/telemetria.  
   - (Opcional) Envie também um **CSV de presets de configuração por cliente**.  
   - No painel:
     - Escolha o **Cliente**;
     - Ajuste/valide a **Configuração desejada** (logoff, tempo, bloqueio, app tempo de direção).

4. O app exibirá:
   - KPIs (total de linhas, placas divergentes, etc.);
   - Tabela apenas com **placas divergentes**;
   - Botão para **baixar o CSV** de divergências.

## 📂 CSV de Presets (opcional)

Se quiser que a configuração seja carregada automaticamente por cliente, use um CSV com colunas:

- `cliente`
- `logoff_enabled` (Ativar/Desativar/1/0/true/false)
- `logoff_seconds` (inteiro, ex.: 30)
- `bloqueio_ignicao` (Ativar/Desativar)
- `app_tempo_direcao` (Ativar/Desativar)

### Exemplo

```csv
cliente,logoff_enabled,logoff_seconds,bloqueio_ignicao,app_tempo_direcao
Cliente A,Ativar,30,Ativar,Desativar
Cliente B,Desativar,0,Desativar,Desativar
```

## 🔧 Ajustes suportados

- Reconhecimento automático de nomes de colunas (com e sem acentos).  
- Suporte a valores booleanos variados: `1/0`, `sim/não`, `true/false`, `ativar/desativar`.  
- Campo de tempo aceita segundos ou formatos `HH:MM:SS` / `MM:SS`.  
- Tolerância configurável para o tempo do logoff.

---
Desenvolvido para facilitar a conferência rápida de configurações de frota 🚛🔎.
