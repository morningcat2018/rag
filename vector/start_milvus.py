from milvus import default_server

if __name__ == "__main__":
    print(f"Milvus Lite 启动中，端口：{default_server.listen_port}")
    # 这行代码会阻塞，表示服务正在运行
    default_server.start()
    print("---")
