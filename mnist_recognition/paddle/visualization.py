from visdom import Visdom

class MyVisual():
    def __init__(self):
        self.viz = Visdom()
        self.loss_acc_list_train = []
        self.loss_acc_list_eval = []
        self.step_list = []
        self.step = 0
        self.epoch_list = []
        self.epoch = 0
        self.opts_train ={
            "title": 'Tain: loss & accuracy',
            "xlabel": 'step',
            "witdh":1200,
            "height":800,
            "legend":['loss', 'accuracy']
        }
        self.opts_eval = {
            "title": 'Evaluate: loss & accuracy',
            "xlabel": 'epoch',
            "witdh": 1200,
            "height": 800,
            "legend": ['loss', 'accuracy']
        }
    def update_train(self, loss, acc):
        self.step += 1
        self.loss_acc_list_train.append([loss, acc])
        self.step_list.append(self.step)
        if self.viz.check_connection():
            self.viz.line(X=self.step_list, Y=self.loss_acc_list_train, win='train', opts=self.opts_train)

    def update_eval(self, loss, acc):
        self.epoch += 1
        self.loss_acc_list_eval.append([loss, acc])
        self.epoch_list.append(self.epoch)
        if self.viz.check_connection():
            self.viz.line(X=self.epoch_list, Y=self.loss_acc_list_eval, win='evaluate', opts=self.opts_eval)




if __name__ == '__main__':
    viz = MyVisual()
    for epoch in range(100):
        for i in range(10):
            v = (epoch)*10+i
            viz.update_train(v, 999-v)
        viz.update_eval(epoch, 99-epoch)
