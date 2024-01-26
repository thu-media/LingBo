import os
d_f = 'rtt_test_trace/'
dirs = os.listdir(d_f)
tm = 100
names = ['12','fcc']
for idx,_trace in enumerate(['12mbps','./fcc/test/test_norway_bus.ljansbakken-oslo-report.2010-09-29_0852CEST.log_720']):
        for f in dirs:
            cmd =' src/experiments/test.py local  --schemes "lingbo" -t 60 --uplink-trace '+_trace+' --downlink-trace '+_trace+' --prepend-mm-cmds "mm-file-delay '+d_f+ f+'  '+str(tm)+'  mm-loss uplink 0.01"   --append-mm-cmds "--uplink-queue=droptail --uplink-queue-args=packets=500" --data-dir '+names[idx]+str(f[:-4])

            os.system(cmd)
        
            cmd = 'src/analysis/analyze.py --data-dir '+names[idx]+str(f[:-4])
    
            os.system(cmd)
                

      
