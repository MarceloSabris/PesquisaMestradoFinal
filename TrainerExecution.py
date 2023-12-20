import tensorflow as tf
import tf_slim as slim
import numpy
import argparse
import os
import random
import time
import Trainer
import sort_of_clevr as DataSetClevr 
from input_ops import create_input_ops
import DataBase as db


def ConfigTrain() : 

   #configurações gerais     
   tf.test.is_gpu_available()
   os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
   parser = argparse.ArgumentParser()
   #caminhos
   parser.add_argument('--datasetPath', type=str, default='Sort-of-CLEVR_teste_decode-image3')
   parser.add_argument('--trainDir', type=str , default='padrao')
   parser.add_argument('--path_restore_train', type=str , default='')
  
   
   #rede de Treinamento
   parser.add_argument('--learning_rate', type=float, default=2.5e-4)
   parser.add_argument('--lr_weight_decay', action='store_true', default=False)
   parser.add_argument('--batch_size', type=int, default=60)
   
  
      
   # Grupos de treinamento , acoes e configurações gerais 
   parser.add_argument('--is_loadImage', type=str , default=False)
   parser.add_argument('--runGenerateDQN', type=int, default='0')

   
   parser.add_argument('--QtdRunAction', type=str, default='100500')
   parser.add_argument('--Actions', type=str, default='3')
   parser.add_argument('--QtdRunTrain', type=int, default='2')
   
    
    
     
     
   config = parser.parse_args()
   
   if type(config.is_loadImage) == type(True): 
       config.is_loadImage = config.is_loadImage
   else:
       if config.is_loadImage == 'True' : 
            config.is_loadImage = True 
       else:
            config.is_loadImage = False
   

    #Variaveis globais config 
   config.path =   path = os.path.join('./datasets', config.datasetPath  ) 

    #variaveis para controle de execução tempo 
   config.tempogravarlog  =0                            
   config.stepTimeTotalExecution = 0  
   config._start_time_total = time.time()
   config.step_time_test_Total = 0
   config.totalTempoGravarArquivoLog = 0
   config.TotalTempoGravaRede =0 
   config._tempoPorRodada = time.time()

    #variaveis para controle de execução tempo 
   config.stepControl = 0   
   config.accuracy_test= 0.0
   config.trainPosition =0
   config.trainGlobal =0
   config.inserirdatabase = True
   config.StepChangeGroupRun=0
   config.trainGlobal = 0
   config.trainPosition = 0
   config.acoes = ["0.1,0.1,0.1,1,1","1,1,1,0.1,0.1","0.5,0.5,0.5,0.1,0.1","0.1,0.1,0.1,0.5,0.5","1,1,1,1,1"] 
   config.QtdLogSave = 1500
   config.GroupQuestion ='naoAplicado' #grupo de questão para rodar até total aprendizado
    #criando o dataset de tereinamento para todos os cenários     
   config.DataSetClevr =  DataSetClevr
   config.dataset_test= config.DataSetClevr.create_default_splits(config.path,is_full =True,id_filename="id_test.txt",is_loadImage=config.is_loadImage)
   config.data_info = DataSetClevr.get_data_info()
   config.conv_info = DataSetClevr.get_conv_info()
   _ , config.batch_train,config.imgs = create_input_ops(config.dataset_test,config.batch_size,
                                               is_training=True,is_loadImage=config.is_loadImage)
    
    
   return config   

def GernerateTrainner (config,trainer): 
    del trainer 
    tf.keras.backend.clear_session()
    tf.compat.v1.reset_default_graph()
    import gc
    trainer = None 
    gc.collect()
    trainer =  Trainer.Trainer(config,config.DataSetClevr,
                      config.dataset_test, config.dataset_test)  
    return trainer  

def RumTrainnerEscolheAcao(config): 

   trainer = ""
   num_actions = 5
   config.QtdRunAction = 3 
   #4
   for quantityExec in range(6) : 
        print ("**************** Execao {}".format(quantityExec)) 
        Acoes=[]
        UsaRand= True
        posicaoAcao = 0
        data = time.strftime(r"%d%m%Y_%H%M", time.localtime())
        for train in range(int(config.QtdRunAction)): 
            config = ConfigTrain()
            config.trainDir = config.trainDir +'_'+ data + "_exec_" + str(train)
            trainer = GernerateTrainner(config,trainer)
            print(" ************  Execution Nr {} ".format(
                            train ))
            
            if train > 0 :
               UsaRand = False
            for stepAction in range(67): 
                config.tipoEscolha="automatica"
                if UsaRand == True : 
                    action = random.randint(0,num_actions-1) 
                    Acoes.append(action)
                    db.add_new_acao(config.trainDir,action)
                    config.acao = action
                else: 
                    action = Acoes[stepAction]  
                    config.acao = action
                config.StepChangeGroupRun = 1500 
                config.GrupDataset =  config.acoes[int(action)] 
                config.dataset_train = config.DataSetClevr.create_default_splits_perc(config.path,is_full =True,grupoDatasets=config.GrupDataset,is_loadImage=config.is_loadImage)
                config.PercDatasetTrain =config.GrupDataset
                trainer.train(config) 

def RumTrainner(config): 

   
   trainer = ""
   qtdsActionRun = config.QtdRunAction.split('|')
   actions = config.Actions.split('|')
   for train in range(config.QtdRunTrain): 
        config = ConfigTrain()
        config.trainDir = config.trainDir +"_Acao_"+ config.Actions.replace('|','_') + "_exec_" + str(train)
        trainer = GernerateTrainner(config,trainer)
        print(" ************  Execution Nr {} ".format(
                            train ))
        
        for action in actions: 
            config.tipoEscolha="manual"
            config.acao = action
            
            config.StepChangeGroupRun = int(qtdsActionRun[min (int(action),len(qtdsActionRun)-1 )])
            config.GrupDataset =  config.acoes[int(action)] 
            config.dataset_train = config.DataSetClevr.create_default_splits_perc(config.path,is_full =True,grupoDatasets=config.GrupDataset,is_loadImage=config.is_loadImage)
            config.PercDatasetTrain =config.GrupDataset
            trainer.train(config)   
                    
def RumManual(config): 

   config.QtdRunAction = [100500]
   config.Actions = [3]
   config.Exec = 1
   config.trainDir = config.trainDir +"_Acao_"+ str(config.Actions[0])+ "_exec_" + str(config.Exec)
   trainer = ""
   qtdsActionRun = config.QtdRunAction
   actions = config.Actions
   trainer = GernerateTrainner(config,trainer)
   for action in actions: 
            config.tipoEscolha="manual"
            config.acao = action
            
            config.StepChangeGroupRun = int(qtdsActionRun[min (int(action),len(qtdsActionRun)-1 )])
            config.GrupDataset =  config.acoes[int(action)] 
            config.dataset_train = config.DataSetClevr.create_default_splits_perc(config.path,is_full =True,grupoDatasets=config.GrupDataset,is_loadImage=config.is_loadImage)
            config.PercDatasetTrain =config.GrupDataset
            trainer.train(config)   
                        

if __name__ == '__main__':
    config = ConfigTrain()
    RumTrainnerEscolheAcao(config) 

   