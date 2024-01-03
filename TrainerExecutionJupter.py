import tensorflow as tf
import tf_slim as slim
import numpy as np
import argparse
import os
import random
import time
import Trainer
import sort_of_clevr as DataSetClevr 
from input_ops import create_input_ops
import DataBase as db
import easydict 



def ConfigTrainJupter(): 
   config =  easydict.EasyDict({
    "batch_size": 100,
    "train_steps": 1000
    })
   #configurações gerais     
   tf.test.is_gpu_available()
   os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
             
   config.datasetPath =  'Sort-of-CLEVR_teste_decode-image'
   config.trainDir =  'TesteDQN'
   config.path_restore_train =  ''
   config.learning_rate = 2.5e-4
   config.lr_weight_decay = False
   config.batch_size = 100
   config.is_loadImage = False
   config.runGenerateDQN = 0
   config.QtdRunAction = 100500
   config.Actions = 5
   config.QtdRunTrain = 1
   config.is_loadImage = False 
   
   config.path =  os.path.join('./datasets', config.datasetPath  ) 

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
  
   config.acoes =  ["0.1,0.1,0.1,1,1","0.5,0.5,0.5,1,1","1,1,1,1,1","1,1,1,0.5,0.5","1,1,1,0.1,0.1"]
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
   config.QtdRunAction = 1
   #4
   for quantityExec in range(20) : 
        print ("**************** Execao {}".format(quantityExec)) 
        Acoes=[]
        UsaRand= True
        posicaoAcao = 0
        data = time.strftime(r"%d%m%Y_%H%M", time.localtime())
        #int(config.QtdRunAction)
        for train in range(1): 
            config = ConfigTrain()
            config.trainDir = config.trainDir +'_'+ str( quantityExec) + '_' + data + "_exec_" + str(train)
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
   config.trainDir = config.trainDir + "_Acao_"+ str(config.Actions[0])+ "_exec_" + str(config.Exec)
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
                        

def RunAction( acao ,config,trainer):
   
   
   config.QtdRunAction = [1500]
   config.Actions = acao
   config.Exec = 1
   qtdsActionRun = config.QtdRunAction
   config.tipoEscolha="DQN"
   config.acao = acao
   config.StepChangeGroupRun = int(qtdsActionRun[min (int(acao),len(qtdsActionRun)-1 )])
   config.GrupDataset =  config.acoes[int(acao)] 
   config.dataset_train = config.DataSetClevr.create_default_splits_perc(config.path,is_full =True,grupoDatasets=config.GrupDataset,is_loadImage=config.is_loadImage)
   config.PercDatasetTrain =config.GrupDataset
   return trainer.train(config)  

def runDQN (episodes):
    state = [0,0,0,0,0]
    config = ConfigTrainJupter()
    trainer = ""
    trainer = GernerateTrainner(config,trainer)    
    for i_episode in range(episodes):
        localhost_save_option = tf.saved_model.SaveOptions(experimental_io_device="/job:localhost")
        
       
        model = tf.keras.models.load_model('D:/source/PesquisaMestradoFinal/interaction_20/',options=localhost_save_option)
       
       
        model = tf.keras.models.load_model('Execucao_10\model_0\interaction_'+str(i_episode))
        action = np.argmax(model.predict(np.array([state]))[0])
        
        state_new = RunAction(action ,config )
        state = [state_new[0],state_new[1],state_new[2],state_new[3],state_new[4]]  


   