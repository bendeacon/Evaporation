import sys
import example.runDPK_invitro as rDI

case = str(sys.argv[1])
N_PROCESS = int(sys.argv[2])
vehicle = str(sys.argv[3])

cfg_fn = 'example/config/CosEU/' + case + '_CE_VH=' + vehicle +'.cfg'
wk_path = '5.Surrey2D_KscVar/' + case + 'VH=' + vehicle + '/'

rDI.compDPK_KwVar(cfg_fn, wk_path=wk_path, N_PROCESS=N_PROCESS)