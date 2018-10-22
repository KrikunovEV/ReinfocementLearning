import visdom

vis = visdom.Visdom()

for episode in range(100):
    vis.line(episode**2, episode, update='append', win='reward')