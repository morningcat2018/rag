from pymilvus import MilvusClient, connections

if __name__ == "__main__":
    connections.connect(host='localhost', port='19530')
    print("Milvus Lite 连接成功！")

    # 连接到你已经启动的独立 Milvus 服务
    # 注意：URI 格式是 "http://host:port"
    client = MilvusClient(uri="http://localhost:19530")
    print("✅ 连接到已运行的 Milvus 服务成功！")

    # 测试连接
    collections = client.list_collections()
    print(f"现有集合: {collections}")

    for cc in collections:
        client.drop_collection(cc)


