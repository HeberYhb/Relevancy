from pykafka import KafkaClient

client = KafkaClient(hosts="192.168.1.10:2181,192.168.1.10:2182,192.168.1.10:2183")
topic = client.topics['newlog2']

consumer=topic.get_balanced_consumer(
    zookeeper_connect='192.168.1.10:2181,192.168.1.10:2182,192.168.1.10:2183'
)

for message in consumer:
    if message is not None:
        print(message.value)