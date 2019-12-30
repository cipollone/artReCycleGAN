''' Utilities '''

import os


class CountersSaver:
  '''\
  This object saves and restores counters: step number and epoch.
  Counters are saved every log_every steps and every epoch.
  '''

  def __init__(self, log_dir, log_every=1):
    '''\
    Args:
      log_dir: directory of logs
      log_every: frequency of savings in number of steps
    '''

    self._log_dir = log_dir
    self._log_every = log_every

    self._filename = os.path.join(log_dir, 'counters.txt')

    # New run
    if not os.path.exists(self._filename):
      self.step = 0
      self.epoch = 0

    # Resuming
    else:
      with open(self._filename) as f:
        lines = f.readlines()
      epoch_s, step_s = lines[1].split(',', 2)
      self.step = int(step_s)
      self.epoch = int(epoch_s)


  def __str__(self):
    return str({'step': self.step, 'epoch': self.epoch})


  def new_step(self):
    ''' Notify one step '''

    self.step += 1

    if self.step % self._log_every == 0:
      self._update_log()


  def new_epoch(self):
    ''' Notify one epoch '''

    self.epoch += 1
    self._update_log()


  def _update_log(self):
    ''' Write the counters to file '''

    with open(self._filename, 'w') as f:
      f.write('epoch, step\n')
      f.write('{}, {}\n'.format(self.epoch, self.step))
