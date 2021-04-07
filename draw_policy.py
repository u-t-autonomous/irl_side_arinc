import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import pickle


def draw_grid(Row,Col):
    fig, ax = plt.subplots()
    x=np.linspace(0,Row,Row+1)
    y=np.linspace(0,Col,Col+1)
    a,b=np.meshgrid(x,y)
    ax.plot(a,b,'k')
    ax.axis('off')
    ax.plot(b,a,'k')
    return fig,ax

def draw_rewards(Row,Col,rewards,fig):
    for state in range(rewards.shape[0]):
        opt=np.argmax(rewards[state,:])
        a0=fig.text(state%Col+0.1,int(state/Col)+0.5,'%0.2f' % rewards[state,0])
        a1=fig.text(state%Col+0.7,int(state/Col)+0.5,'%0.2f' % rewards[state,1])
        a2=fig.text(state%Col+0.4,int(state/Col)+0.8,'%0.2f' % rewards[state,2])
        a3=fig.text(state%Col+0.4,int(state/Col)+0.1,'%0.2f' % rewards[state,3])
        if opt == 0:
            a0.set_color('r')
        elif opt == 1:
            a1.set_color('r')
        elif opt == 2:
            a2.set_color('r')
        elif opt == 3:
            a3.set_color('r')
    return fig

def draw_incentive_direction(Row,Col,incentive_direction,fig):
    for state in range(incentive_direction.shape[0]):
        if (state - state%(Row*Col))/(Row*Col)==0 and incentive_direction[state] == 0:
            fig.arrow((state%(Row*Col))%Col+0.4,int((state%(Row*Col))/Col)+0.7,-0.2,0,head_width=0.1, head_length=0.1)
        if (state - state%(Row*Col))/(Row*Col)==1 and incentive_direction[state] == 0:
            fig.arrow((state%(Row*Col))%Col+0.8,int((state%(Row*Col))/Col)+0.7,-0.2,0,head_width=0.1, head_length=0.1, color = 'r')
        if (state - state%(Row*Col))/(Row*Col)==2 and incentive_direction[state] == 0:
            fig.arrow((state%(Row*Col))%Col+0.4,int((state%(Row*Col))/Col)+0.3,-0.1,0,head_width=0.05, head_length=0.05)
        if (state - state%(Row*Col))/(Row*Col)==3 and incentive_direction[state] == 0:
            fig.arrow((state%(Row*Col))%Col+0.8,int((state%(Row*Col))/Col)+0.3,-0.1,0,head_width=0.05, head_length=0.05, color = 'r')

        if (state - state%(Row*Col))/(Row*Col)==0 and incentive_direction[state] == 1:
            fig.arrow((state%(Row*Col))%Col+0.1,int((state%(Row*Col))/Col)+0.7, 0.2,0,head_width=0.1, head_length=0.1)
        if (state - state%(Row*Col))/(Row*Col)==1 and incentive_direction[state] == 1:
            fig.arrow((state%(Row*Col))%Col+0.5,int((state%(Row*Col))/Col)+0.7, 0.2,0,head_width=0.1, head_length=0.1, color = 'r')
        if (state - state%(Row*Col))/(Row*Col)==2 and incentive_direction[state] == 1:
            fig.arrow((state%(Row*Col))%Col+0.1,int((state%(Row*Col))/Col)+0.3, 0.1,0,head_width=0.05, head_length=0.05)
        if (state - state%(Row*Col))/(Row*Col)==3 and incentive_direction[state] == 1:
            fig.arrow((state%(Row*Col))%Col+0.5,int((state%(Row*Col))/Col)+0.3, 0.1,0,head_width=0.05, head_length=0.05, color = 'r')

        if (state - state%(Row*Col))/(Row*Col)==0 and incentive_direction[state] == 2:
            fig.arrow((state%(Row*Col))%Col+0.2,int((state%(Row*Col))/Col)+0.6, 0,0.2,head_width=0.1, head_length=0.1)
        if (state - state%(Row*Col))/(Row*Col)==1 and incentive_direction[state] == 2:
            fig.arrow((state%(Row*Col))%Col+0.6,int((state%(Row*Col))/Col)+0.6, 0,0.2,head_width=0.1, head_length=0.1, color = 'r')
        if (state - state%(Row*Col))/(Row*Col)==2 and incentive_direction[state] == 2:
            fig.arrow((state%(Row*Col))%Col+0.2,int((state%(Row*Col))/Col)+0.2, 0,0.1,head_width=0.05, head_length=0.05)
        if (state - state%(Row*Col))/(Row*Col)==3 and incentive_direction[state] == 2:
            fig.arrow((state%(Row*Col))%Col+0.6,int((state%(Row*Col))/Col)+0.2, 0,0.1,head_width=0.05, head_length=0.05, color = 'r')

        if (state - state%(Row*Col))/(Row*Col)==0 and incentive_direction[state] == 3:
            fig.arrow((state%(Row*Col))%Col+0.2,int((state%(Row*Col))/Col)+0.8, 0,-0.2,head_width=0.1, head_length=0.1)
        if (state - state%(Row*Col))/(Row*Col)==1 and incentive_direction[state] == 3:
            fig.arrow((state%(Row*Col))%Col+0.6,int((state%(Row*Col))/Col)+0.8, 0,-0.2,head_width=0.1, head_length=0.1, color = 'r')
        if (state - state%(Row*Col))/(Row*Col)==2 and incentive_direction[state] == 3:
            fig.arrow((state%(Row*Col))%Col+0.2,int((state%(Row*Col))/Col)+0.4, 0,-0.1,head_width=0.05, head_length=0.05)
        if (state - state%(Row*Col))/(Row*Col)==3 and incentive_direction[state] == 3:
            fig.arrow((state%(Row*Col))%Col+0.6,int((state%(Row*Col))/Col)+0.4, 0,-0.1,head_width=0.05, head_length=0.05, color = 'r')
    return fig

def draw_incentive_direction2(Row,Col,incentive_direction,fig):
    # for i in range(Row):
    #     for j in range(Col):
    #         if np.argmax(incentive_direction[i,j,:]) == 3:
    #             fig.arrow(i+0.4,j+0.5,-0.2,0,head_width=0.1, head_length=0.1)
    #         elif np.argmax(incentive_direction[i,j,:]) == 2:
    #             fig.arrow(i+0.1,j+0.5, 0.2,0,head_width=0.1, head_length=0.1)
    #         elif np.argmax(incentive_direction[i,j,:]) == 0:
    #             fig.arrow(i+0.2,j+0.4, 0,0.2,head_width=0.1, head_length=0.1)
    #         elif np.argmax(incentive_direction[i,j,:]) == 1:
    #             fig.arrow(i+0.2,j+0.6, 0,-0.2,head_width=0.1, head_length=0.1)
    #         if np.argsort(incentive_direction[i,j,:])[-2] == 3:
    #             fig.arrow(i+0.8,j+0.5,-0.1,0,head_width=0.05, head_length=0.05, color = 'r')
    #         elif np.argsort(incentive_direction[i,j,:])[-2] == 2:
    #             fig.arrow(i+0.5,j+0.5, 0.1,0,head_width=0.05, head_length=0.05, color = 'r')
    #         elif np.argsort(incentive_direction[i,j,:])[-2] == 0:
    #             fig.arrow(i+0.6,j+0.4, 0,0.1,head_width=0.05, head_length=0.05, color = 'r')
    #         elif np.argsort(incentive_direction[i,j,:])[-2] == 1:
    #             fig.arrow(i+0.6,j+0.6, 0,-0.1,head_width=0.05, head_length=0.05, color = 'r')
    for i in range(Row):
        for j in range(Col):
            if np.argmax(incentive_direction[i,j,:]) == 3:
                fig.arrow(j+0.5,Row-i-0.5,-0.2,0,head_width=0.1, head_length=0.1)
            elif np.argmax(incentive_direction[i,j,:]) == 2:
                fig.arrow(j+0.3,Row-i-0.5, 0.2,0,head_width=0.1, head_length=0.1)
            elif np.argmax(incentive_direction[i,j,:]) == 0:
                fig.arrow(j+0.4,Row-i-0.6, 0,0.2,head_width=0.1, head_length=0.1)
            elif np.argmax(incentive_direction[i,j,:]) == 1:
                fig.arrow(j+0.4,Row-i-0.4, 0,-0.2,head_width=0.1, head_length=0.1)
            if np.argsort(incentive_direction[i,j,:])[-2] == 3:
                fig.arrow(j+0.8,Row-i-0.5,-0.1,0,head_width=0.05, head_length=0.05, color = 'r')
            elif np.argsort(incentive_direction[i,j,:])[-2] == 2:
                fig.arrow(j+0.6,Row-i-0.5, 0.1,0,head_width=0.05, head_length=0.05, color = 'r')
            elif np.argsort(incentive_direction[i,j,:])[-2] == 0:
                fig.arrow(j+0.7,Row-i-0.6, 0,0.1,head_width=0.05, head_length=0.05, color = 'r')
            elif np.argsort(incentive_direction[i,j,:])[-2] == 1:
                fig.arrow(j+0.7,Row-i-0.4, 0,-0.1,head_width=0.05, head_length=0.05, color = 'r')
    # for state in range(incentive_direction.shape[0]):
    #     if (state - state%(Row*Col))/(Row*Col)==0 and incentive_direction[state] == 0:
    #         fig.arrow((state%(Row*Col))%Col+0.4,int((state%(Row*Col))/Col)+0.7,-0.2,0,head_width=0.1, head_length=0.1)
    #     if (state - state%(Row*Col))/(Row*Col)==1 and incentive_direction[state] == 0:
    #         fig.arrow((state%(Row*Col))%Col+0.8,int((state%(Row*Col))/Col)+0.7,-0.2,0,head_width=0.1, head_length=0.1, color = 'r')
    #     if (state - state%(Row*Col))/(Row*Col)==2 and incentive_direction[state] == 0:
    #         fig.arrow((state%(Row*Col))%Col+0.4,int((state%(Row*Col))/Col)+0.3,-0.1,0,head_width=0.05, head_length=0.05)
    #     if (state - state%(Row*Col))/(Row*Col)==3 and incentive_direction[state] == 0:
    #         fig.arrow((state%(Row*Col))%Col+0.8,int((state%(Row*Col))/Col)+0.3,-0.1,0,head_width=0.05, head_length=0.05, color = 'r')
    #
    #     if (state - state%(Row*Col))/(Row*Col)==0 and incentive_direction[state] == 1:
    #         fig.arrow((state%(Row*Col))%Col+0.1,int((state%(Row*Col))/Col)+0.7, 0.2,0,head_width=0.1, head_length=0.1)
    #     if (state - state%(Row*Col))/(Row*Col)==1 and incentive_direction[state] == 1:
    #         fig.arrow((state%(Row*Col))%Col+0.5,int((state%(Row*Col))/Col)+0.7, 0.2,0,head_width=0.1, head_length=0.1, color = 'r')
    #     if (state - state%(Row*Col))/(Row*Col)==2 and incentive_direction[state] == 1:
    #         fig.arrow((state%(Row*Col))%Col+0.1,int((state%(Row*Col))/Col)+0.3, 0.1,0,head_width=0.05, head_length=0.05)
    #     if (state - state%(Row*Col))/(Row*Col)==3 and incentive_direction[state] == 1:
    #         fig.arrow((state%(Row*Col))%Col+0.5,int((state%(Row*Col))/Col)+0.3, 0.1,0,head_width=0.05, head_length=0.05, color = 'r')
    #
    #     if (state - state%(Row*Col))/(Row*Col)==0 and incentive_direction[state] == 2:
    #         fig.arrow((state%(Row*Col))%Col+0.2,int((state%(Row*Col))/Col)+0.6, 0,0.2,head_width=0.1, head_length=0.1)
    #     if (state - state%(Row*Col))/(Row*Col)==1 and incentive_direction[state] == 2:
    #         fig.arrow((state%(Row*Col))%Col+0.6,int((state%(Row*Col))/Col)+0.6, 0,0.2,head_width=0.1, head_length=0.1, color = 'r')
    #     if (state - state%(Row*Col))/(Row*Col)==2 and incentive_direction[state] == 2:
    #         fig.arrow((state%(Row*Col))%Col+0.2,int((state%(Row*Col))/Col)+0.2, 0,0.1,head_width=0.05, head_length=0.05)
    #     if (state - state%(Row*Col))/(Row*Col)==3 and incentive_direction[state] == 2:
    #         fig.arrow((state%(Row*Col))%Col+0.6,int((state%(Row*Col))/Col)+0.2, 0,0.1,head_width=0.05, head_length=0.05, color = 'r')
    #
    #     if (state - state%(Row*Col))/(Row*Col)==0 and incentive_direction[state] == 3:
    #         fig.arrow((state%(Row*Col))%Col+0.2,int((state%(Row*Col))/Col)+0.8, 0,-0.2,head_width=0.1, head_length=0.1)
    #     if (state - state%(Row*Col))/(Row*Col)==1 and incentive_direction[state] == 3:
    #         fig.arrow((state%(Row*Col))%Col+0.6,int((state%(Row*Col))/Col)+0.8, 0,-0.2,head_width=0.1, head_length=0.1, color = 'r')
    #     if (state - state%(Row*Col))/(Row*Col)==2 and incentive_direction[state] == 3:
    #         fig.arrow((state%(Row*Col))%Col+0.2,int((state%(Row*Col))/Col)+0.4, 0,-0.1,head_width=0.05, head_length=0.05)
    #     if (state - state%(Row*Col))/(Row*Col)==3 and incentive_direction[state] == 3:
    #         fig.arrow((state%(Row*Col))%Col+0.6,int((state%(Row*Col))/Col)+0.4, 0,-0.1,head_width=0.05, head_length=0.05, color = 'r')
    return fig

def draw_specific_arrow(Row,Col,state,incentive_direction,fig):
    if incentive_direction == 0:
        fig.arrow(state%Col+0.8,int(state/Col)+0.4,-0.4,0,head_width=0.1, head_length=0.1)
    if incentive_direction == 1:
        fig.arrow(state%Col+0.4,int(state/Col)+0.4, 0.4,0,head_width=0.1, head_length=0.1)
    if incentive_direction== 2:
        fig.arrow(state%Col+0.5,int(state/Col)+0.2, 0,0.4,head_width=0.1, head_length=0.1)
    if incentive_direction == 3:
        fig.arrow(state%Col+0.4,int(state/Col)+0.8, 0,-0.4,head_width=0.1, head_length=0.1)
    return fig

if __name__ == "__main__":
    with open("pi_theta_12_corner_100.pkl","rb") as f:
        mdp_objs_tr = pickle.load(f)
    print(np.argmax(mdp_objs_tr[0,5,0,:]))

    a1,ax1=draw_grid(12,12)
    ax1=draw_incentive_direction2(12,12,mdp_objs_tr[:,:,0,:],ax1)
    plt.show(ax1)
