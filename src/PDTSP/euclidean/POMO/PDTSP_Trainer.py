import torch
from logging import getLogger

from PDTSP_Env import PDTSPEuclideanEnv as Env
from PDTSP_Model import PDTSPModel as Model

from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler

from utils import *

class PDTSPTrainer:
    def __init__(self, env_params, model_params, optimizer_params, trainer_params):
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()

        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type(torch.cuda.FloatTensor)
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type(torch.FloatTensor)

        self.model = Model(**self.model_params)
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **optimizer_params['scheduler'])

        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.last_epoch = model_load['epoch'] - 1
            self.logger.info('Saved mPDTSPModel Loaded!')

        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset(self.start_epoch)
        epochs = self.trainer_params['epochs']
        for epoch in range(self.start_epoch, epochs + 1):
            self.logger.info('-------------------------------------------------')
            train_score, train_loss = self._train_one_epoch(epoch)
            self.scheduler.step()
            self.result_log.append('train_score', epoch, train_score)
            self.result_log.append('train_loss', epoch, train_loss)

            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, epochs)
            self.logger.info("Epoch {:3d}/{:3d}: Elapsed[{}], Remain[{}]".format(
                epoch, epochs, elapsed_time_str, remain_time_str))

            all_done = (epoch == epochs)
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            img_save_interval = self.trainer_params['logging']['img_save_interval']

            if epoch > 1:
                image_prefix = '{}/latest'.format(self.result_folder)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                               self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                               self.result_log, labels=['train_loss'])

            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info('Saving model...')
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            if all_done or (epoch % img_save_interval) == 0:
                image_prefix = '{}/img/checkpoint-{}'.format(self.result_folder, epoch)
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_1'],
                                               self.result_log, labels=['train_score'])
                util_save_log_image_with_label(image_prefix, self.trainer_params['logging']['log_image_params_2'],
                                               self.result_log, labels=['train_loss'])

            if all_done:
                self.logger.info('Training completed!')
                util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):
        score = AverageMeter()
        loss = AverageMeter()
        train_episode_num = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0
        while episode < train_episode_num:
            remaining  = train_episode_num - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)
            avg_score, avg_loss = self._train_one_batch(batch_size)
            score.update(avg_score, batch_size)
            loss.update(avg_loss, batch_size)
            episode += batch_size

            if epoch == self.start_epoch and loop_cnt < 10:
                loop_cnt += 1
                self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:.1f}%) | Score: {:.4f} | Loss: {:.4f}'
                    .format(epoch, episode, train_episode_num, 100 * episode/train_episode_num,
                    score.avg, loss.avg))

        self.logger.info('Epoch {:3d}: Score: {:.4f} | Loss: {:.4f}'
                         .format(epoch, score.avg, loss.avg))

        return score.avg, loss.avg

    def _train_one_batch(self, batch_size):
        self.model.train()
        pomo_size = self.env.pomo_size

        self.env.load_problems(batch_size)
        reset_state, _, _ = self.env.reset()
        self.model.pre_forward(reset_state)

        prob_list = torch.zeros(batch_size, pomo_size, 0)
        state, reward, done = self.env.pre_step()
        while not done:
            selected, prob = self.model(state)
            state, reward, done = self.env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)
        loss = -advantage * log_prob
        # shape: (batch, pomo)
        loss_mean = loss.mean()

        # Score
        ###############################################
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value

        # Step & Return
        ###############################################
        self.model.zero_grad()
        loss_mean.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return score_mean.item(), loss_mean.item()