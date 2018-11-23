# -*- coding: utf-8 -*-
"""
Created on Tue Nov 08 18:54:27 2016

@author: BobD
"""
import visdom
import numpy as np


class Visualizer(object):
    def __init__(self, env, title, labels):
        assert(type(labels) == list)
        self.labels = labels
        self.env = env
        self.numoflabels = len(labels)
        self.vis = visdom.Visdom(env=env, port=8030)
        self.opts = dict(title=title, legend=labels)
        self.iteration = 0
        self.vis_line = None
        self.title = title

    def __call__(self, epoch_id, *args):
        assert(len(args) == self.numoflabels)
        self.iteration = epoch_id
        if self.vis_line is None:
            self.vis_line = self.vis.line(X=np.array([[self.iteration] * len(args)]), Y=np.array([args]),
                                          win=self.title, opts=self.opts)
        else:
            self.vis.line(X=np.array([[self.iteration] * len(args)]), Y=np.array([args]), opts=self.opts, win=self.vis_line,
                          update='replace' if self.iteration == 0 else 'append')

    def image(self, arr, title, win=1):
        self.vis.image(arr, win=win, opts=dict(title=title))
        
    def text(self, s, title, win=2):
        self.vis.text(s, win=win, opts=dict(title=title))

    def is_running(self):
        return self.vis.check_connection()

    def save(self):
        self.vis.save(envs=[self.env])
