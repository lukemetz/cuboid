from multiprocessing import Process
from fuel.streams import ServerDataStream
from subprocess import Popen

def get_open_port():
    # http://stackoverflow.com/questions/2838244/get-open-tcp-port-in-python
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return port

hwm = 50


def on_thread(make_datastream, port):
    import fuel.server
    fuel.server.logger.setLevel("WARN")
    stream = make_datastream()
    fuel.server.start_server(stream, port, hwm)


def fork_to_background(make_datastream, sources):
    port = get_open_port()
    proc = Process(target=on_thread, args=(make_datastream, port))
    proc.start()
    datastream = ServerDataStream(sources,
                                  port=port,
                                  hwm=hwm,
                                  produces_examples=False)
    return datastream, proc

def stream_from_file(filename, *args):
    port = get_open_port()
    proc = Popen(['python', filename, str(port)] + args, env=dict(os.environ, THEANO_FLAGS='device=cpu'))
    stream = ServerDataStream(ss.sources, port=port, hwm=50, produces_examples=False)
    return stream, proc
