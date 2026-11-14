"""Generate editable SVG network block diagrams using only the standard library."""
from pathlib import Path
from html import escape

DEST = Path(__file__).resolve().parents[1] / 'docs/figures'
NAVY='#25435e'; BLUE='#275d85'; TEAL='#247f79'; ORANGE='#bb733c'; GREY='#6b7d8d'


class SVG:
    def __init__(self,width,height,title,description):
        self.w,self.h=width,height
        self.items=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
                    f'<title id="title">{escape(title)}</title><desc id="desc">{escape(description)}</desc>',
                    '<defs>'+''.join(f'<marker id="a{i}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse"><path d="M 0 1 L 9 5 L 0 9 Z" fill="{color}"/></marker>' for i,color in enumerate((NAVY,TEAL,ORANGE,GREY)))+'</defs>',
                    '<rect width="100%" height="100%" fill="white"/>']

    def text(self,x,y,value,size=24,color=NAVY,bold=False,anchor='middle'):
        self.items.append(f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{color}" font-family="Microsoft YaHei, Noto Sans CJK SC, PingFang SC, sans-serif" font-size="{size}" font-weight="{600 if bold else 400}">{escape(value)}</text>')

    def rect(self,x,y,w,h,fill='white',stroke=BLUE,dashed=False,r=5):
        self.items.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="2"'+(' stroke-dasharray="10 7"' if dashed else '')+'/>')

    def line(self,points,color=NAVY,dashed=False,arrow=True,width=2.4):
        marker=(NAVY,TEAL,ORANGE,GREY).index(color)
        self.items.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in points)+f'" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round"'+(' stroke-dasharray="8 6"' if dashed else '')+(f' marker-end="url(#a{marker})"' if arrow else '')+'/>')

    def dot(self,x,y,color=NAVY):
        self.items.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{color}"/>')

    def op(self,x,y,label):
        self.items.append(f'<circle cx="{x}" cy="{y}" r="24" fill="white" stroke="{NAVY}" stroke-width="2.4"/>')
        self.text(x,y+9,label,30)

    def block(self,x,y,w,h,lines,color=BLUE,fill='#f1f6fa'):
        self.rect(x,y,w,h,fill,color)
        first=y+h/2-(len(lines)-1)*17+8
        for i,label in enumerate(lines):self.text(x+w/2,first+i*34,label,25 if i==0 else 21,color,i==0)

    def save(self,name):
        DEST.mkdir(parents=True,exist_ok=True)
        (DEST/name).write_text('\n'.join(self.items+['</svg>']),encoding='utf-8',newline='\n')


def training():
    s=SVG(1800,660,'CSI JSCM 联合训练网络框图','UE 编码器、前缀与功率归一化、固定 QAM、AWGN、补零与 BS 解码器；重建损失联合训练两端网络。')
    s.text(45,48,'CSI JSCM 联合训练网络',30,bold=True,anchor='start')
    s.rect(155,100,1500,325,fill='none',stroke=ORANGE,dashed=True,r=8)
    s.items.append('<rect x="715" y="84" width="380" height="32" fill="white"/>')
    s.text(905,108,'端到端联合训练范围',25,ORANGE,True)
    s.rect(180,145,640,240,'#fbfdfe','#c4d4df')
    s.rect(850,145,200,240,'#fafbfc','#d6dde3')
    s.rect(1080,145,550,240,'#fbfdfc','#c5d9d3')
    s.text(205,180,'UE 发送端',25,BLUE,True,'start')
    s.text(950,180,'反馈信道',25,GREY,True)
    s.text(1105,180,'BS 接收端',25,TEAL,True,'start')
    s.block(205,215,190,95,['编码器','CNN + AF'])
    s.block(425,215,170,95,['前缀选择','功率归一化'])
    s.block(625,215,170,95,['固定 16-QAM','STE 近似梯度'])
    s.block(875,215,150,95,['AWGN','实际功率定噪'],GREY,'#f3f5f7')
    s.block(1105,215,145,95,['接收补零','128 实维'],TEAL,'#eff7f4')
    s.block(1285,215,320,95,['解码器','FC + AF + 5 个残差块'],TEAL,'#eff7f4')
    for a,b in ((130,205),(395,425),(595,625),(795,875),(1025,1105),(1250,1285),(1605,1685)):
        s.line([(a,262),(b,262)])
    s.text(78,273,'H',40,bold=True)
    s.text(78,313,'2 × 32 × 32',19,GREY)
    s.text(1730,273,'Ĥ',40,bold=True)
    s.text(1730,313,'重建 CSI',21,GREY)
    s.text(500,355,'SNR 条件输入 AF；Ns ∈ {16, 32, 64}',22,BLUE)
    s.text(1355,355,'SNR 条件输入 AF',22,TEAL)
    s.block(775,480,250,85,['重建损失','L(Ĥ, H)'],ORANGE,'#fff7ef')
    s.line([(78,332),(78,522),(775,522)])
    s.line([(1730,332),(1730,522),(1025,522)])
    s.text(395,505,'训练参考 H',22,GREY)
    s.text(1375,505,'重建 Ĥ',22,GREY)
    s.line([(900,480),(900,426)],ORANGE,True)
    s.text(930,459,'反向传播',21,ORANGE,anchor='start')
    s.text(900,618,'重建损失共同更新编码器与解码器；固定星座和信道无可训练参数。',24,GREY)
    s.save('jscm_joint_training.svg')


def receiver():
    s=SVG(1800,700,'BS 质量监测与共享残差精修网络框图','共享解码器输出基础重建和中间特征，特征与 SNR 用于预测质量；门控按需调用共享精修器，最终输出为基础重建加门控残差。')
    s.text(45,48,'BS 接收端：质量监测与按需精修',30,bold=True,anchor='start')
    s.block(210,240,260,120,['共享解码器','基础重建 + 中间特征'])
    s.text(65,310,'y',37,bold=True)
    s.line([(95,300),(210,300)])
    s.block(650,115,270,100,['质量预测头','池化特征 f + SNR'],TEAL,'#eff7f4')
    s.block(980,115,240,100,['门控模块','g = 1[q̂ > τ]'],ORANGE,'#fff7ef')
    s.line([(340,240),(340,165),(650,165)],TEAL)
    s.text(460,148,'中间特征 f',22,TEAL)
    s.dot(550,165,TEAL)
    s.line([(920,165),(980,165)],TEAL)
    s.text(950,145,'q̂',24,TEAL)
    s.line([(1220,165),(1290,165),(1290,476)],ORANGE,True)
    s.text(1320,220,'门控 g',22,ORANGE,anchor='start')
    s.block(660,430,420,140,['共享残差精修器','y 投影 + 特征拼接','3 层 CNN 输出 ΔH'],TEAL,'#eff7f4')
    s.line([(470,300),(1436,300)])
    s.text(1050,285,'基础重建 Ĥ',24,BLUE)
    s.dot(585,300)
    s.line([(585,300),(585,500),(660,500)])
    s.text(615,485,'Ĥ',22,BLUE)
    s.line([(550,165),(550,450),(660,450)],TEAL)
    s.text(615,436,'f',23,TEAL)
    s.dot(145,300)
    s.line([(145,300),(145,610),(810,610),(810,570)])
    s.text(460,596,'接收观测 y',22,BLUE)
    s.line([(1080,500),(1266,500)],TEAL)
    s.text(1170,483,'修正量 ΔH',23,TEAL)
    s.op(1290,500,'×')
    s.line([(1314,500),(1460,500),(1460,324)])
    s.op(1460,300,'+')
    s.line([(1484,300),(1655,300)])
    s.text(1720,310,'Ĥout',36,bold=True)
    s.text(1720,350,'最终重建',22,GREY)
    s.text(1450,550,'g · ΔH',24,TEAL)
    s.text(900,664,'Ĥout = Ĥ + g · ΔH。门限由验证集固定，仅 g = 1 时调用精修器。',25,GREY)
    s.save('bs_quality_refinement.svg')


if __name__=='__main__':
    training();receiver()
    print('Generated docs/figures/*.svg')
