#!/usr/bin/env python

from os import path
from subprocess import check_call

import arg_parser
import context


def main():
    args = arg_parser.receiver_first()
    # hard-code
    
    LingBo_path = 'yourpath/test'

    send_src = path.join(LingBo_path, 'run_sender.py')

    recv_src = path.join(LingBo_path,  'run_receiver.py')


    model_file =  path.join(LingBo_path, 'model', 'nn_model_ep_2600.ckpt') 

    queue_flag = '0.3'
    periodic_flag = 0
    if args.option == 'setup':
        check_call(['sudo pip install tensorflow==1.14.0 tflearn==0.5'], shell=True)
        return

    if args.option == 'receiver':
        cmd = ['python', recv_src, args.port]
        check_call(cmd)
        return

    if args.option == 'sender':
        cmd = ['python', send_src, args.ip, args.port, model_file, queue_flag, periodic_flag]
        check_call(cmd)
        return


if __name__ == '__main__':
    main()
