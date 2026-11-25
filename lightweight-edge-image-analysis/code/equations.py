"""把公式渲染为透明 PNG，供 Word 插入。"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({'mathtext.fontset': 'stix', 'font.family': 'STIXGeneral'})

EQ = {
 'eq1': r'$L(\theta)=\frac{1}{N}\sum_{i=1}^{N}\ell\left(f_{\theta}(x_i),\,y_i\right)$',
 'eq2': r'$y_{i,j,k}^{(l)}=f\left(\sum_{c}\sum_{u,v}w_{u,v,c,k}^{(l)}\,x_{i+u,j+v,c}^{(l-1)}+b_{k}^{(l)}\right)$',
 'eq3': r'$\mathrm{Attn}(Q,K,V)=\mathrm{softmax}\,(QK^{T}/\sqrt{d_k})\,V$',
 'eq4': r'$\mathcal{L}=(1-\alpha)\,\frac{T^{2}}{N}\sum_{i}\mathrm{KL}(p_i^{T}\,\|\,q_i^{T})+\alpha\,\frac{1}{N}\sum_{i}\mathrm{CE}(y_i,\,q_i)$',
 'eq5': r'$q=\mathrm{clamp}(\mathrm{round}(w/s)+z,\;0,\;255),\quad s=(w_{\max}-w_{\min})/255$',
}


def render(name, tex, fontsize=15):
    fig = plt.figure(figsize=(0.01, 0.01))
    t = fig.text(0, 0, tex, fontsize=fontsize)
    fig.savefig(f'../figures/{name}.png', dpi=400, transparent=True,
                bbox_inches='tight', pad_inches=0.04)
    plt.close(fig)


if __name__ == '__main__':
    for k, v in EQ.items():
        render(k, v)
    print('equations done')
