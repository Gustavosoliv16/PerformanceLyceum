import requests
import os
import json
# 1. Configure sua chave API
API_KEY = os.getenv('LYCEUM_API_KEY', 'lk_88859c8946be789815c313f5b85f274442af421c1c99fb498a1f44791317509a')

url = "https://api.lyceum.technology/api/v2/external/execution/start"

# 3. Headers com autenticação Bearer
headers = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json"
}

# 4. Seu código Python de teste de GPU
codigo_python = '''
import torch
import torchvision.models as models
import time

# --- Configuração ---
# Verifica se a GPU está disponível
if not torch.cuda.is_available():
    print("Erro: GPU (CUDA) não disponível. Saindo.")
    exit()

device = torch.device("cuda")
print(f"Usando GPU: {torch.cuda.get_device_name(0)}")

# --- Passo 1: Medir o tempo de carregamento do modelo ---
print("\\n--- Teste 1: Carregamento do Modelo ---")
start_load_time = time.time()

# Carrega um modelo ResNet-50 pré-treinado
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
# Coloca o modelo em modo de avaliação (importante para inferência)
model.eval()
# *** O passo crucial: envia o modelo para a GPU ***
model.to(device)

torch.cuda.synchronize() # Garante que a operação terminou
end_load_time = time.time()
print(f"Tempo para carregar o modelo na GPU: {end_load_time - start_load_time:.4f} segundos.")

# --- Passo 2: Medir o tempo de inferência ---
print("\\n--- Teste 2: Performance de Inferência ---")
# Simula um "batch" de 64 imagens de tamanho 224x224 com 3 canais de cor (RGB)
# Este é um tamanho de entrada padrão para o ResNet
dummy_input = torch.randn(64, 3, 224, 224, device=device)

# Aquece a GPU para medições precisas
print("Aquecendo a GPU para inferência...")
with torch.no_grad(): # Desativa o cálculo de gradientes para acelerar
    model(dummy_input)

# Mede a performance
print("Iniciando medição de inferência...")
start_inference_time = time.time()

with torch.no_grad():
    # Executa a inferência 100 vezes para obter uma média
    for _ in range(100):
        _ = model(dummy_input)

torch.cuda.synchronize() # Garante que todas as operações terminaram
end_inference_time = time.time()

total_time = end_inference_time - start_inference_time
images_processed = 64 * 100
fps = images_processed / total_time

print("Medição finalizada.")
print(f"Tempo para 100 execuções: {total_time:.4f} segundos.")
print(f"Performance: {fps:.2f} imagens por segundo (FPS).")
'''

# 5. Payload da requisição com dependências PyTorch
payload = {
    "code": codigo_python,
    "timeout": 300,  # 5 minutos - este script pode demorar por causa do download do modelo
    "requirements_content": "torch\ntorchvision"  # Dependências necessárias
}

# 6. Fazer a requisição
print("Enviando script para execução na Lyceum Cloud...")
print("Isso pode demorar alguns minutos devido ao tamanho do modelo ResNet-50...\n")

try:
    response = requests.get(url, headers=headers, json=payload)
    response.raise_for_status()
    
    # 7. Processar resposta
    data = response.json()
    
    print("=" * 60)
    print("RESPOSTA DA API LYCEUM")
    print("=" * 60)
    print(json.dumps(data, indent=2))
    
    # Se houver output/logs, exibe formatado
    if 'output' in data:
        print("\n" + "=" * 60)
        print("OUTPUT DO SCRIPT")
        print("=" * 60)
        print(data['output'])
    
    if 'logs' in data:
        print("\n" + "=" * 60)
        print("LOGS")
        print("=" * 60)
        print(data['logs'])
    
except requests.exceptions.HTTPError as e:
    print(f"❌ Erro HTTP: {e}")
    print(f"Status Code: {response.status_code}")
    print(f"Resposta: {response.text}")
    
    # Erros comuns
    if response.status_code == 401:
        print("\n⚠️  Erro de autenticação. Verifique se sua API KEY está correta.")
    elif response.status_code == 403:
        print("\n⚠️  Acesso negado. Verifique as permissões da sua API KEY.")
    elif response.status_code == 429:
        print("\n⚠️  Muitas requisições. Aguarde um momento antes de tentar novamente.")
        
except requests.exceptions.RequestException as e:
    print(f"❌ Erro na requisição: {e}")
except Exception as e:
    print(f"❌ Erro inesperado: {e}")