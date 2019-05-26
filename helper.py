import keras.backend as K


def huber_loss(y_true, y_pred, max_grad=1.):
    diff = y_true - y_pred
    diff_squared = diff * diff
    max_grad_squared = max_grad * max_grad
    return max_grad_squared * (K.sqrt(1. + diff_squared/max_grad_squared) - 1.)


def mean_huber_loss(y_true, y_pred, max_grad=1.):
    loss = huber_loss(y_true, y_pred, max_grad=max_grad)
    return K.mean(loss, axis=None)



def get_frame(agent,villeges,zombies,angle):
    frame=[[4]*5 for _ in range(5)]
    if agent[0]!=-99:
        agent_x,agent_y=transform_state(agent[0],agent[1])
        frame[agent_x][agent_y]=0+angle/180.0
    if villeges[0]!=-99:
        villeges_x,villeges_y=transform_state(villeges[0],villeges[1])
        frame[villeges_x][villeges_y]=2
    if zombies[0]!=-99:
        zombies_x,zombies_y=transform_state(zombies[0],zombies[1])
        frame[zombies_x][zombies_y]=3
    
    return frame


def transform_state(x,y):
    if x>=3:
        x=2.5
    if y>=3:
        y=2.5
    if x<=-2:
        x=-1.5
    if y<=-2:
        y=-1.5
    return int(x+2),int(y+2)