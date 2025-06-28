#Ricardo de Paula Xavier - 2515750
#Leonardo Naime Lima - 2515660

import socket
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np # type: ignore
import time
from threading import Thread, Lock

# Configurações
HOST = '0.0.0.0'
PORT = 2055
WINDOW_SIZE = 15
SENSITIVITY_THRESHOLD = 0.02
MAX_POINTS = 200

# Variáveis compartilhadas (protegidas por Lock)
data_buffer = deque(maxlen=MAX_POINTS)
conn = None
conn_lock = Lock()
is_running = True

# Configuração do gráfico
plt.switch_backend('TkAgg')
fig, ax = plt.subplots()
lines = ax.plot([], [], 'r-', [], [], 'g-', [], [], 'b-', linewidth=0.7)
ax.set_ylim(-1.5, 1.5)
ax.set_xlim(0, MAX_POINTS)
ax.legend(['Eixo X', 'Eixo Y', 'Eixo Z'], loc='upper right')
ax.set_title('Dados do Giroscópio/Acelerômetro (Suavizado)')
plt.tight_layout()

def filter_data(raw_x, raw_y, raw_z):
    avg_x = np.clip(raw_x, -2, 2)
    avg_y = np.clip(raw_y, -2, 2)
    avg_z = np.clip(raw_z, -2, 2)
    
    if abs(avg_x) < SENSITIVITY_THRESHOLD: avg_x = 0
    if abs(avg_y) < SENSITIVITY_THRESHOLD: avg_y = 0
    if abs(avg_z) < SENSITIVITY_THRESHOLD: avg_z = 0
    
    return avg_x, avg_y, avg_z

def socket_server():
    global conn, is_running
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(5)  # Timeout para evitar bloqueio eterno
        s.bind((HOST, PORT))
        s.listen()
        print(f"Aguardando conexão do HyperIMU na porta {PORT}...")
        
        try:
            conn, addr = s.accept()
            with conn_lock:
                print(f"Conectado! Endereço: {addr}")
            
            while is_running:
                try:
                    data = conn.recv(1024).decode().strip()
                    if data and data != "-":
                        x, y, z = map(float, data.split(','))
                        filtered = filter_data(x, y, z)
                        data_buffer.append((time.time(), *filtered))
                except (ValueError, socket.error):
                    continue
                    
        except Exception as e:
            print(f"Erro no servidor: {e}")
        finally:
            is_running = False

def update_graph(frame):
    if not data_buffer:
        return lines
    
    # Extrai dados do buffer
    timestamps = [d[0] for d in data_buffer]
    x_data = [d[1] for d in data_buffer]
    y_data = [d[2] for d in data_buffer]
    z_data = [d[3] for d in data_buffer]
    
    # Atualiza gráfico
    lines[0].set_data(range(len(x_data)), x_data)
    lines[1].set_data(range(len(y_data)), y_data)
    lines[2].set_data(range(len(z_data)), z_data)
    
    # Ajusta eixo X se necessário
    if len(x_data) == MAX_POINTS:
        ax.set_xlim(0, len(x_data))
    
    return lines

# Inicia servidor em thread separada
server_thread = Thread(target=socket_server, daemon=True)
server_thread.start()

# Configura animação
ani = animation.FuncAnimation(
    fig, 
    update_graph, 
    interval=16,  # ~60 FPS
    blit=True,
    cache_frame_data=False
)

try:
    plt.show()
finally:
    is_running = False
    server_thread.join(timeout=1)
    if conn:
        conn.close()
    print("Conexão encerrada.")