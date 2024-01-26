# LingBo

LingBo is a congestion control algorithm published in the WWW24 paper titled "Meet challenges of RTT Jitter, A Hybrid Internet Congestion Control Algorithm".

## Testing with Pantheon

If you would like to test LingBo, please use Pantheon by following these steps:

1. Place `LingBo.py` in the `src/wrappers` directory and add `lingbo` to the `src/config.yml` file. Modify the `LingBo_path` in `LingBo.py` to the location where you store the test folder:

   ```python
   LingBo_path = 'yourpath/test'
   ```

2. The default mahimahi does not support scenarios with variable delay. We recommend installing from the source code at https://github.com/ameya98/mahimahi, as this version supports emulating variable-delay links.

3. We provide some RTT trace files to assist with testing, which can be found in the `rtt_test_trace` directory.

4. You can run the Pantheon-based script to test:

   ```python
   python test_lingbo.py
   ```

