#! /usr/bin/env python
# -*- coding:utf-8 -*-

#from plan import Planner
from predict import Seq2SeqPredictor
import sys

import tensorflow as tf
tf.app.flags.DEFINE_boolean('cangtou', False, 'Generate Acrostic Poem')

reload(sys)
sys.setdefaultencoding('utf8')

def get_cangtou_keywords(input):
    assert(len(input) == 4)
    return [c for c in input]

def main():

    #planner = Planner()
    with Seq2SeqPredictor() as predictor:
        def client_thread(conn, ip, port, MAX_BUFFER_SIZE = 4096):

            # the input is in bytes, so decode it
            input_from_client_bytes = conn.recv(MAX_BUFFER_SIZE)

            # MAX_BUFFER_SIZE is how big the message can be
            # this is test if it's sufficiently big
            import sys
            siz = sys.getsizeof(input_from_client_bytes)
            if  siz >= MAX_BUFFER_SIZE:
                print("The length of input is probably too long: {}".format(siz))

            # decode input and strip the end of line
            input_from_client = input_from_client_bytes.decode("utf8").rstrip()

            res = predictor.predict(input_from_client)
            print("Result of processing {} is: {}".format(input_from_client, res))

            vysl = res.encode("utf8")  # encode the result string
            conn.sendall(vysl)  # send it to client
            conn.close()  # close connection
            print('Connection ' + ip + ':' + port + " ended")

        def start_server():

            import socket
            soc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # this is for easy starting/killing the app
            soc.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            print('Socket created')

            try:
                soc.bind(("127.0.0.1", 12345))
                print('Socket bind complete')
            except socket.error as msg:
                import sys
                print('Bind failed. Error : ' + str(sys.exc_info()))
                sys.exit()

            #Start listening on socket
            soc.listen(10)
            print('Socket now listening')

            # for handling task in separate jobs we need threading
            from threading import Thread

            # this will make an infinite loop needed for
            # not reseting server for every client
            while True:
                conn, addr = soc.accept()
                ip, port = str(addr[0]), str(addr[1])
                print('Accepting connection from ' + ip + ':' + port)
                try:
                    Thread(target=client_thread, args=(conn, ip, port)).start()
                except:
                    print("Terible error!")
                    import traceback
                    traceback.print_exc()
            soc.close()
        start_server()
        # Run loop
        terminate = False
        while not terminate:
            try:
                input = raw_input('Input Text:\n').decode('utf-8').strip()

                if not input:
                    print 'Input cannot be empty!'
                elif input.lower() in ['quit', 'exit']:
                    terminate = True
                else:
                    lines = predictor.predict(input)
                    print lines

            except EOFError:
                terminate = True
            except KeyboardInterrupt:
                terminate = True
    print '\nTerminated.'

if __name__ == '__main__':
    main()
