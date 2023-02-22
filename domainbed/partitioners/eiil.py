from domainbed.partitioners.partitioner import Partitioner
from domainbed.partitioners.partition import Partition
import torch
from torch import optim
from tqdm import tqdm

class EIIL(Partitioner):
    """
    Implementation of the EIIL algorithm, as seen in Environment Inference 
    for Invariant Learning, Creager et al. 2021.
    Required hparams: lr, n_steps, batch_size, num_workers
    """
    def __init__(self, hparams):
        super(EIIL, self).__init__(hparams)

    def split(self, dataset, reference_classifier, test_envs, num_envs, return_history=False):
        self.partition = Partition(dataset)
        # for old_env, values in enumerate(dataset):
        #     for i in range(len(values)):
        #         self.partition.assign(original_environment=old_env, old_index=i, new_environment=old_env)
        logits, y = self._get_logits(dataset, reference_classifier, test_envs)
        scale = torch.tensor(1.).cuda().requires_grad_()
        loss = torch.nn.CrossEntropyLoss(reduction='none')(
          logits.cuda() * scale, 
          y.cuda()
        )
        
        env_w, penalty, history = self._get_environment_distribution(logits, scale, loss, num_envs)
        self._generate_mapping(dataset, test_envs, env_w)
        if return_history:
          return self.partition, history
        return self.partition

    def _get_logits(self, dataset, reference_classifier, test_envs):
      """Return all logits and ys for non test datasets"""
      batch_size, num_workers = self.hparams['batch_size'], self.hparams['num_workers']
      logits, all_y = [], []
      for i, env in enumerate(dataset):
        if i in test_envs:
          continue
        batch_dataloader = torch.utils.data.DataLoader(env, 
          batch_size=batch_size,
          shuffle=False, 
          num_workers=num_workers
        )
        for x, y in batch_dataloader:
          logits.append(reference_classifier.predict(x.cuda()).detach())
          all_y.append(y)
      logits = torch.cat(logits)
      all_y = torch.cat(all_y)
      return logits, all_y
    
    def _get_environment_distribution(self, logits, scale, loss, num_envs):
      lr, n_steps = self.hparams['lr'], self.hparams['n_steps']
      env_w = torch.randn((len(logits), num_envs)).cuda().requires_grad_()
      optimizer = optim.Adam([env_w], lr=lr)
      no_tqdm=False
      losses = [ ]
      with tqdm(total=n_steps, disable=no_tqdm) as desc:
          for i in tqdm(range(n_steps), disable=no_tqdm):
            penalties = [ ]
            probs = torch.divide(env_w.sigmoid().T, env_w.sigmoid().sum(axis=1))
            for env_index in range(num_envs):
              loss_env = (loss.squeeze() * probs[env_index]).mean()
              grad_env = torch.autograd.grad(loss_env, [scale], create_graph=True)[0]
              penalties.append(torch.sum(grad_env**2))
            npenalty = - torch.stack(penalties).mean()
            losses.append(npenalty.cpu().item())

            optimizer.zero_grad()
            npenalty.backward(retain_graph=True)
            optimizer.step()
      return env_w.detach().cpu(), npenalty.cpu().item(), losses
    
    def _generate_mapping(self, dataset, test_envs, env_w):
      new_environments = torch.argmax(env_w.sigmoid().T / env_w.sigmoid().sum(axis=1), axis=0)
      k = 0
      for i, env in enumerate(dataset):
        if i in test_envs:
          continue
        for j in range(len(env)):
          new_env = int(new_environments[k].cpu().numpy()) 
          k += 1
          self.partition.assign(original_environment=i, old_index=j, new_environment=new_env)        

