import time
import sys
import socket
import select
import numpy as np
import datagram_pb2
import network as network
READ_FLAGS = select.POLLIN | select.POLLPRI
WRITE_FLAGS = select.POLLOUT
ERR_FLAGS = select.POLLERR | select.POLLHUP | select.POLLNVAL
READ_ERR_FLAGS = READ_FLAGS | ERR_FLAGS
ALL_FLAGS = READ_FLAGS | WRITE_FLAGS | ERR_FLAGS

S_INFO = 9
S_LEN = 8
ACTIONS = [-1., 1.] # , 0.8, 1.6]
A_DIM = 1
MAX_STEP = 500
MAX_TIME_OUT = 30
MAX_LOSS = 100

class VideoSender(object):

    def curr_ts_ms(self):
        if self.init_ts is None:
            self.init_ts = time.time()

        return int((time.time() - self.init_ts) * 1000)
    def   check_get_rtt(self,action,duration_):
        get_rtt_thresholds = 3 * self.min_rtt_std / duration_ * 1.5
        if get_rtt_thresholds > action:
            self.get_rtt_state = False
        return self.get_rtt_state 
    def get_95_min_rtt(self,rtt_list):
        rtt_5 = np.percentile(rtt_list,5)
        rtt_95 = np.percentile(rtt_list,95)
        new_list = []
        for item in rtt_list:
            if rtt_5 < rtt_95 and  (item < int(rtt_5) or item > int(rtt_95+0.5)):
                continue
            else:
                new_list.append(item)
        if len(new_list) == 0:
            new_list = rtt_list
        return np.mean(new_list), np.std(new_list)
    def __init__(self, ip, port, model='./model/model.ckpt',queue_flag=0.3, periodic_flag=0):

        self.init_ts = None
        self.model_path = model
        self.actor = network.Network([S_INFO, S_LEN], A_DIM, 1e-4)
        self.actor.load_model(self.model_path)
        # UDP socket and poller
        self.peer_addr = (ip, port)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        
        self.poller = select.poll()
        self.poller.register(self.sock, ALL_FLAGS)

        self.dummy_payload = 'x' * 1400

        # congestion control related
        self.seq_num = 0
        self.next_ack = 0
        self.cwnd = 2.0
        self.step_len_ms = 100
        self.min_loss = None

        self.init_state = True
        self.judge_state = False
        self.tmp_rtts = []
        self.get_rtt_state = True
        self.last_send_time = 0

        # state variables for RLCC
        self.delivered_time = 0
        self.delivered = 0
        self.sent_bytes = 0

        self.min_rtt = float('inf')
        self.min_rtt_std = None
        self.min_rtts = []
        self.delay_ewma = None
        self.send_rate_ewma = None
        self.delivery_rate_ewma = None
        self.flag = 0.5
        
        self.step_start_ms = None
        self.step_loss_ms = None

        self.running = True
        self.state = np.zeros([S_INFO, S_LEN])

        action_prob = self.actor.predict(self.state)

        self.step_cnt = 0
        self.done = False
        self.start_time = None

        self.ts_first = None
        self.rtt_buf = []
        self.thr_buff = []
        self.loss_buff = []
        self.loss = 0.
        self.srtt = 0.


        self.curr_bytes, self.curr_delay = [], []
        self.loss_count=0
        self.queue_flag = float(queue_flag)
        self.periodic_flag = periodic_flag 


    def cleanup(self):
        self.sock.close()

    def handshake(self):
        self.sock.setblocking(0)  # non-blocking UDP socket

    def loss_ratio(self):
  
        self.loss_buff.sort()

        cnt = 0
        while True:
            now_ =  self.loss_buff[-1][0]
            send_ts_, seq_num_ = self.loss_buff[cnt]
            if now_ - send_ts_ > MAX_LOSS:
                cnt += 1
            else:
                break
        self.loss_buff = self.loss_buff[cnt:]
        
        tmp_loss = []
        for t in self.loss_buff:
            send_ts_, seq_num_ = t
            tmp_loss.append(seq_num_)
        
 
        if len(tmp_loss) > 0:
            loss_count = len(tmp_loss)
            loss_sum = tmp_loss[-1] - tmp_loss[0] + 1
            loss_ = 1. - loss_count / float(loss_sum)
        else:
            loss_ = 0.

        if self.min_loss is None:
            self.min_loss = loss_

        self.min_loss = np.min([self.min_loss, loss_])
        return loss_
        

    def update_state(self, ack):
        """ Update the state variables listed in __init__() """
        #self.min_loss = 0#self.loss_ratio(ack)
        self.loss_buff.append([ack.send_ts, ack.seq_num])
        self.next_ack = max(self.next_ack, ack.seq_num + 1)
        curr_time_ms = self.curr_ts_ms()

        # Update RTT
        rtt = float(curr_time_ms - ack.send_ts)
        if  self.get_rtt_state and not self.judge_state:
            self.min_rtts.append(rtt)

            self.min_rtt,self.min_rtt_std = self.get_95_min_rtt(self.min_rtts)
       

        if self.judge_state and self.get_rtt_state :
            if ack.send_ts < self.last_send_time:
                self.min_rtts.append(rtt)
            else:
                self.tmp_rtts.append(rtt)
        if not self.get_rtt_state:
            if ack.send_ts < self.last_send_time:
                self.min_rtts.append(rtt)
                self.min_rtt,self.min_rtt_std = self.get_95_min_rtt(self.min_rtts)
              
           
        if len(self.min_rtts) >= 4 and  self.init_state:
            self.init_state = False
           

        self.srtt = rtt

        if self.ts_first is None:
            self.ts_first = curr_time_ms
        self.rtt_buf.append(rtt)

        delay = rtt - self.min_rtt
        
        if self.delay_ewma is None:
            self.delay_ewma = delay
        else:
            self.delay_ewma = 0.875 * self.delay_ewma + 0.125 * delay

        # Update BBR's delivery rate
        self.delivered += ack.ack_bytes
        self.delivered_time = curr_time_ms
        delivery_rate = (0.008 * (self.delivered - ack.delivered) /
                         max(1, self.delivered_time - ack.delivered_time))

        if self.delivery_rate_ewma is None:
            self.delivery_rate_ewma = delivery_rate
        else:
            self.delivery_rate_ewma = (
                0.875 * self.delivery_rate_ewma + 0.125 * delivery_rate)

        self.curr_bytes.append(ack.ack_bytes)
        self.curr_delay.append(delay)
        # Update Vegas sending rate
        send_rate = 0.008 * (self.sent_bytes - ack.sent_bytes) / max(1, rtt)

        if self.send_rate_ewma is None:
            self.send_rate_ewma = send_rate
        else:
            self.send_rate_ewma = (
                0.875 * self.send_rate_ewma + 0.125 * send_rate)

    def take_action(self, action_idx):
        action_idx = np.clip(action_idx, -1., 1.)
        self.cwnd = (1. + action_idx) * self.cwnd
        self.cwnd = np.clip(self.cwnd, 2.0, 5000.0)

    def window_is_open(self):
        return self.seq_num - self.next_ack < self.cwnd

    def send(self):
        # sys.stderr.write('sending ' + str(self.seq_num) + '\n')
        data = datagram_pb2.Data()
        data.seq_num = self.seq_num
        data.send_ts = self.curr_ts_ms()
        data.sent_bytes = self.sent_bytes
        data.delivered_time = self.delivered_time
        data.delivered = self.delivered
        data.payload = self.dummy_payload

        serialized_data = data.SerializeToString()
        self.sock.sendto(serialized_data, self.peer_addr)

        self.seq_num += 1
        self.sent_bytes += len(serialized_data)

    def step(self, current_state, duration):
        state = np.roll(self.state, -1, axis=1)
        for p in range(S_INFO):
            state[p, -1] = current_state[p]

        action_prob = self.actor.predict(state)
        act = action_prob[0]

        # compute reward
        cum_bytes = np.sum(self.curr_bytes)
        # bytes / 1000 / ms -> kb / ms -> mb / s
        _throughput = 0.008 * cum_bytes / (duration + 1e-6)
        if len(self.curr_delay) > 0:
            _delay = np.mean(self.curr_delay)
        else:
            _delay = 0.
        
        # clean curr_buff
        self.curr_bytes = []
        self.curr_delay = []

        self.state = state

        return self.state, act, action_prob

    def recv(self, max_steps = 300):
        serialized_ack, addr = self.sock.recvfrom(1600)

        if addr != self.peer_addr:
            return

        ack = datagram_pb2.Ack()

        ack.ParseFromString(serialized_ack)

        self.update_state(ack)

        if self.step_start_ms is None:
            self.step_start_ms = self.curr_ts_ms()
            self.step_loss_ms = self.curr_ts_ms()

        # At each step end, feed the state:
        duration_ = self.curr_ts_ms() - self.step_start_ms

        if duration_ > self.step_len_ms and not self.init_state:  # step's end
            self.judge_state = True
            self.loss = self.loss_ratio()
        
            if self.get_rtt_state:
                self.min_rtt,self.min_rtt_std = self.get_95_min_rtt(self.min_rtts)
              
            s = [self.delay_ewma / 100.,
                    self.delivery_rate_ewma / 20.,
                    self.send_rate_ewma / 20.,
                    self.loss,
                    self.cwnd / 500.,
                    duration_ / 100.,
                    self.min_rtt/100.,
                    self.min_rtt_std/10.,
                    self.queue_flag]
            
  
            state, action, prob = self.step(s, duration_)
            if self.get_rtt_state:
                
                if self.check_get_rtt(action,duration_):
                    self.last_send_time = self.curr_ts_ms()
                    self.min_rtts += self.tmp_rtts
                    self.tmp_rtts = []
                  

                  
                else:
                    if len(self.tmp_rtts) > 0:
                        self.min_rtts.append(self.tmp_rtts[0])
                        
                self.min_rtt,self.min_rtt_std = self.get_95_min_rtt(self.min_rtts)
              
            
                s = [self.delay_ewma / 100.,
                        self.delivery_rate_ewma / 20.,
                        self.send_rate_ewma / 20.,
                        self.loss,
                        self.cwnd / 500.,
                        duration_ / 100.,
                        self.min_rtt/100.,
                        self.min_rtt_std/10.,
                        self.queue_flag]

                state, action, prob = self.step(s, duration_)
                    
               


            self.take_action(action)

           
            self.delay_ewma = None
            self.delivery_rate_ewma = None
            self.send_rate_ewma = None

            self.step_start_ms = self.curr_ts_ms()
            
            self.step_cnt += 1
            
        

            curr_ms = self.curr_ts_ms()
            if curr_ms - self.start_time >= 600. * 1000.:
                # if self.delivered > maximum_send_mb * 1000. * 1000.:
                self.step_cnt = 0
                self.running = False
                self.done = True

        if self.periodic_flag and self.curr_ts_ms() - self.last_send_time  > 10*1000 and self.get_rtt_state == False:
         
            self.init_state = True
            self.judge_state = False
            self.tmp_rtts = []
            self.min_rtts = []
            self.get_rtt_state = True
            self.cwnd = self.cwnd * (self.min_rtt/(2*self.min_rtt+3*self.min_rtt_std))
            self.last_send_time = self.curr_ts_ms()

    def run(self):
        VIDEO_CHUNK_COUNT = 1
        MAX_SEND_CHUNK_SIZE = 1.
        cum_delivered = 0
        for p in range(VIDEO_CHUNK_COUNT):
            # state variables for RLCC
            self.delivered_time = 0
            self.delivered = 0
            self.sent_bytes = 0
            self.running = True
            self.step_cnt = 0
            self.done = False

            
            self.single_step(5000, 30.)
            cum_delivered += self.delivered


    def single_step(self, max_steps=500, time_out=30.):
        TIMEOUT = 1000

        self.poller.modify(self.sock, ALL_FLAGS)
        curr_flags = ALL_FLAGS

        self.start_time = self.curr_ts_ms()
        while self.running:

            if self.window_is_open():
                if curr_flags != ALL_FLAGS:
                    self.poller.modify(self.sock, ALL_FLAGS)
                    curr_flags = ALL_FLAGS
            else:
                if curr_flags != READ_ERR_FLAGS:
                    self.poller.modify(self.sock, READ_ERR_FLAGS)
                    curr_flags = READ_ERR_FLAGS

            events = self.poller.poll(TIMEOUT)

            if not events:  # timed out
                self.send()

            for fd, flag in events:
                assert self.sock.fileno() == fd

                if flag & ERR_FLAGS:
                    sys.exit('Error occurred to the channel')

                if flag & READ_FLAGS:
                    self.recv(max_steps)

                if flag & WRITE_FLAGS:
                    if self.window_is_open():
                        self.send()




    
