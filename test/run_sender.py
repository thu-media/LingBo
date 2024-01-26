import argparse
import numpy as np
from os import path
from video_sender import VideoSender

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('ip', metavar='IP')
    parser.add_argument('port', type=int)
    parser.add_argument('model', type=str)
    parser.add_argument('queue_flag', type=float)
    parser.add_argument('periodic_flag', type=float)
    args = parser.parse_args()
    
    sender = VideoSender(args.ip, args.port, args.model,args.queue_flag, args.periodic_flag)

    try:
        sender.handshake()
        sender.run()
    except KeyboardInterrupt:
        pass
    finally:
        sender.cleanup()


if __name__ == '__main__':
    main()
    exit(0)
