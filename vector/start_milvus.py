from milvus import default_server
import threading
import time


def start_milvus_server():
    """在后台线程中启动 Milvus Lite"""
    print(f"启动 Milvus Lite，端口：{default_server.listen_port}")
    default_server.start() # 数据存储在 ~/.milvus-io/milvus-server

    # 等待服务完全启动
    time.sleep(2)

    if default_server.running:
        print(f"✅ Milvus Lite 服务已启动")
        print(f"   地址: localhost:{default_server.listen_port}")
    else:
        print("❌ 启动失败")


if __name__ == "__main__":
    # 在后台线程中启动（不阻塞主线程）
    server_thread = threading.Thread(target=start_milvus_server, daemon=True)
    server_thread.start()

    # 主线程可以继续做其他事情
    print("主线程继续运行...")

    # 保持程序运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n停止 Milvus Lite...")
        default_server.stop()

    """
    # 检查端口是否被监听
    lsof -i:19530
    
    # 或者使用 netcat 测试连接
    nc -zv localhost 19530
    """