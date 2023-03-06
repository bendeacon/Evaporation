import sys
import example.runDPK_invitro as rDI

case = str(sys.argv[1])
N_PROCESS = int(sys.argv[2])

#case = args[1]
#print(case)
cfg_fn = 'example/config/CosEU/' + case + '_CE.cfg'
wk_path = '5.Surrey2D_KscVar/' + case + '/'

rDI.compDPK_KwVar(cfg_fn, wk_path=wk_path, N_PROCESS=N_PROCESS)