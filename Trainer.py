
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
#import plaidml.keras
import os
#plaidml.keras.install_backend()

import csv
import matplotlib.pyplot as plt
from util import log

from pprint import pprint

from input_ops import create_input_ops

from vqa_util import NUM_COLOR
from vqa_util import visualize_iqa,question2str,answer2str

import time
import tensorflow as tf
import tf_slim as slim
import json

from datetime import datetime

import DataBase as Postgree
from datetime import datetime
from collections import deque
import sort_of_clevr as DataSetClevr 

import tensorflow as tf 


class Trainer(object):

    @staticmethod
    
    def get_model_class():
        from model_rn import Model
        return Model

    
    def __init__(self,
                 config,
                 data,
                 dataset,
                 dataset_test):
        
       # hyper_parameter_str = config.datasetPath+'_lr_'+str(config.learning_rate)
        #self.trainDir = './trainDir/%s-%s-%s-%s' % (
        #    config.model,
        #    config.prefix,
        #    hyper_parameter_str,
        #    time.strftime("%Y%m%d-%H%M%S")
        #)
        self.trainDir = './trainDir/%s' % (config.trainDir)

        if not os.path.exists(self.trainDir):
            os.makedirs(self.trainDir)
        log.infov("Train Dir: %s", self.trainDir)
        
        # --- input ops ---
        self.batch_size = config.batch_size
        self.data = data
        _, self.batch_train,imgs = create_input_ops(dataset, self.batch_size,shuffle=False,
                                               is_training=True,is_loadImage= config.is_loadImage)
       
            
        _, self.batch_test,ims = create_input_ops(dataset_test, self.batch_size,shuffle=False,                                          
                                              is_training=False,is_loadImage= config.is_loadImage)
        self.DataSetPath = os.path.join('./datasets', config.datasetPath)
        # --- create model ---
        Model = self.get_model_class()
        self.model = Model(config)
        
        #tf.keras.utils.plot_model(self.model, show_shapes=True, show_layer_names=True)
        # --- optimizer ---
        self.global_step = tf.compat.v1.train.get_or_create_global_step(graph=None)
        self.learning_rate = config.learning_rate
        if config.lr_weight_decay:
            self.learning_rate = tf.train.exponential_decay(
                self.learning_rate,
                global_step=self.global_step,
                decay_steps=10000,
                decay_rate=0.5,
                staircase=True,
                name='decaying_learning_rate'
            )
        self.QtdTest = len(dataset_test) 
        self.check_op = tf.no_op()
        self.trainPosition = 0 
        self._ids = []
        self._predictions = []
        self._groundtruths = []
        self._questions = [] 
        self._answers=[]
        self._images = [] 
        self._predictionsTrain = []
        self._groundtruthsTrain = []
        self._questionsTrain = [] 
        self._answersTrain=[]
        self._imagesTrain = [] 
        
        self.ArrayQuestoesCertas = [] 
        self.ArrarQuestoesErradas =[]
        #self.check_pathSaveTrain= config.trainDir
        self.ArrayTotalQuestoesCertas =[0,0,0,0,0]
        self.ArrayTotalQuestoesErradas =[0,0,0,0,0]
       

        self.testAcuracy = 0 
        self.optimizer = slim.optimize_loss(
            loss=self.model.loss,
            global_step=self.global_step,
            learning_rate=self.learning_rate,
            optimizer='Adam',
            clip_gradients=20.0,
            name='optimizer_loss'
        )
       
        self.summary_op = tf.compat.v1.summary.merge_all()
        #salvar
        #import tfplot
        #self.plot_summary_op =  tf.compat.v1.summary.merge_all()
        #salvar 
        #self.saver = tf.compat.v1.train.Saver(max_to_keep=10000)
        #self.summary_writer = tf.compat.v1.summary.FileWriter(self.trainDir)
    
        
        self.pathDataSets = os.path.join('./datasets', config.datasetPath)
        self.checkpoint_secs = 60000  # 10 min
        self.acuracy = [] 
        self.step = []
        #savar arquivo
        self.supervisor = tf.compat.v1.train.Supervisor(
        #      logdir=self.trainDir,
               is_chief=True,
              saver=None,
              summary_op=None,
        #    summary_writer=self.summary_writer, salvar
        #    save_summaries_secs=3000,
        
        #    save_model_secs=self.checkpoint_secs,
            global_step=self.global_step,
        )
        tf.test.is_gpu_available()

        #gpu_config = tf.GPUOptions()
        #gpu_config.visible_device_list = "1"

        session_config = tf.compat.v1.ConfigProto(
            allow_soft_placement=True,
             intra_op_parallelism_threads=16,
             inter_op_parallelism_threads=16,
            gpu_options=tf.compat.v1.GPUOptions(allow_growth=True),
            device_count={'GPU': 2},
        )
        #self.tf_config.gpu_options.allow_growth=True
        #gavar arquivo
        self.session = self.supervisor.prepare_or_wait_for_session(config=session_config)
        self.session.graph._unsafe_unfinalize()

        self.ckpt_path = config.path_restore_train
      
        if len(self.ckpt_path) > 0 :
            log.info("Checkpoint path: %s", self.ckpt_path)
            self.saver.restore(self.session, self.ckpt_path)
            log.info("Loaded the pretrain parameters from the provided checkpoint path")

    


    def GravarArquivo1 ( self,data_dict,fname):
      
       print("gravar arquivo: " + fname + " qtd: " +  str(len(data_dict)))
       os.makedirs(self.trainDir, exist_ok=True)
       fname = self.trainDir + "/" + fname +".json"
       # Create file
       with open(fname, 'a') as outfile:
         json.dump(data_dict, outfile, ensure_ascii=False, indent=2) 
         outfile.write('\n')
         outfile.close() 

    def add_batch (self, id,questions,ans, prediction, groundtruth):
        # for now, store them all (as a list of minibatch chunks)
        self._ids.append(id)
        self._questions.append(questions)
        self._answers.append(ans)
        self._predictions.append(prediction)
        self._groundtruths.append(groundtruth)

    
    def plot_acuracy(self):
            plt.plot(self.step,self.acuracy)
            
            #plt.ylim([0, 10])
            plt.xlabel('step')
            plt.ylabel('Cauracy')
            plt.legend()
            plt.grid(True)
            plt.savefig(   self.trainDir + '/acuracy_datalenght.png')
         #   plt.show()
    def GravarTreino(self,trainGlobal,trainDir ,accuracy,accuracy_test,questaotipo0,questaotipo1, questaotipo2,questaotipo3,questaotipo4,PercDatasetTrain,tipoEscolha,acao,reward,totalreard):
        Postgree.add_new_row(trainGlobal,trainDir ,accuracy,accuracy_test,questaotipo0,questaotipo1, questaotipo2,questaotipo3,questaotipo4,PercDatasetTrain,tipoEscolha,acao,reward,totalreard  )
   
    def train(self,config):
        log.infov("Training Starts!")
      
        #alterei aqui 
        accuracy = 0.00
            #accuracy_total = 0.001
         
        _start_time_total = time.time()
        s=1
        stepExecution = 0
        _, batch_train,imgs = create_input_ops(config.dataset_train, config.batch_size,
                                               is_training=True,is_loadImage=config.is_loadImage)
        
        stepTimeTotalExecution = 0
        step_time_test_Total = 0 
        questaotipo0 =0
        questaotipo1 = 0    
        questaotipo2 = 0
        questaotipo3 = 0
        questaotipo4 = 0
        totalTempoGravarArquivoLog =0
        TotalTempoGravaRede = 0 
     
        
        
        while  stepExecution <= int(config.StepChangeGroupRun)  : 
          #datasetcont = -1
          #for cont in self.config.orderDataset.split(',') :
         
          step, accuracy, summary, loss, step_time = \
                    self.run_single_step(batch_train, step=config.trainGlobal,config=config)
          stepTimeTotalExecution = step_time + stepTimeTotalExecution
                #accuracy_total = accuracy +accuracy_total 1
          #or  stepExecution == config.StepChangeGroupRun
          if config.trainGlobal % config.QtdLogSave == 0  :

                 # periodic inference
                    accuracy_test, step_time_test,questaotipo0,questaotipo1,questaotipo2,questaotipo3,questaotipo4 = \
                        self.run_test('Teste' ,step,config.dataset_test)
                    step_time_test_Total = step_time_test + step_time_test_Total
                    self.log_step_message(step, accuracy, accuracy_test, loss, step_time,step_time_test)
                    tempogravarlog = time.time()
                    
                    temp=[]
                    temp.append( str( config.trainGlobal))
                    temp.append(  str(config.trainDir) )
                    temp.append(str(accuracy))
                    temp.append(str(accuracy_test))
                    temp.append(str(questaotipo0))
                    temp.append(str(questaotipo1))
                    temp.append(str(questaotipo2))
                    temp.append(str(questaotipo3))
                    temp.append(str(questaotipo4))    
                    temp.append(str(config.PercDatasetTrain))
                    temp.append(str(config.tipoEscolha))
                    temp.append(str(config.acao))
                    temp.append(str(0))
                    temp.append(str(0))
                    self.GravarCSV(temp,'Logs')
                    
                    temp=[]
                    temp.append('step:'+ str( config.trainGlobal))
                    temp.append('accuracy:'+ str(accuracy))
                    temp.append('accuracy_test:'+ str(accuracy_test))
                    temp.append('questaotipo0:'+str(questaotipo0))
                    temp.append('questaotipo1:'+str(questaotipo1))
                    temp.append('questaotipo2:'+str(questaotipo2))
                    temp.append('questaotipo3:'+str(questaotipo3))
                    temp.append('questaotipo4:'+str(questaotipo4))  
                    temp.append('PercDatasetTrain:'+str(config.PercDatasetTrain))           
                    self.GravarArquivo1(temp,'Logs')
                    config.accuracy  = accuracy
                    config.accuracy_test = accuracy_test
                    if config.inserirdatabase == True :
                        Postgree.add_new_row(config.trainGlobal,config.trainDir ,accuracy,accuracy_test,questaotipo0,questaotipo1, questaotipo2,questaotipo3,questaotipo4,config.PercDatasetTrain,config.tipoEscolha,config.acao ,0,0)
                    else: 
                        config.passoInserir = config.trainGlobal 
                       
                    totalTempoGravarArquivoLog = totalTempoGravarArquivoLog + (time.time() -  tempogravarlog)
                    
                   #salvar
                   # self.summary_writer.add_summary(summary, global_step=config.trainGlobal )         

                #log.infov( 'Tempo total treinamentoada' + str((time.time() - _tempoPorRodada)))
                    now = datetime.now()
                #self.run_test('Treino' ,step,self.dataset)
                    current_time = now.strftime("%H:%M:%S")

                    inicioTempoGravaRede = time.time()
                    log.infov( "%s Saved checkpoint at %d",config.trainGlobal,  s) 
                    #salvar
                    #try: 
                    #   save_path = self.saver.save(self.session,
                    #                        os.path.join(self.trainDir, 'model'),
                    #                        global_step=step)
                    #except Exception as error:
                    #   print('****************** erro -- ao executar ***********')
                    #   print(error) 
                    
                    TotalTempoGravaRede = TotalTempoGravaRede + (time.time() -inicioTempoGravaRede )
          if ( stepExecution >1 and  stepExecution%(10*config.dataset_train.maxGrups)  == 0): 
                 
                 self.data.updateIdsDataSet( os.path.join('./datasets', config.datasetPath  ), config.dataset_train, grupoDatasets= config.PercDatasetTrain)
                 _, self.batch_train,imgs = create_input_ops(config.dataset_train, config.batch_size,
                                               is_training=True,is_loadImage=config.is_loadImage)
          if  ( config.GroupQuestion=="NoRelational" and   questaotipo0>0.98  and questaotipo1 >0.98 and questaotipo2>0.98   ) : 
               stepExecution = stepExecution+1     
               config.trainPosition = config.trainPosition + 1      
               config.trainGlobal = config.trainGlobal +1
               break;
          if  ( config.GroupQuestion=="Relational" and   questaotipo3 >0.90 and questaotipo4>0.90   ) : 
               stepExecution = stepExecution+1     
               config.trainPosition = config.trainPosition + 1      
               config.trainGlobal = config.trainGlobal +1
               break;
          
          
          
          #config.trainPosition = config.trainPosition+1  
          stepExecution = stepExecution+1     
          config.trainPosition = config.trainPosition + 1      
          config.trainGlobal = config.trainGlobal +1
        #salvar arquivo
        #self.plot_acuracy()
        _end_time_total = time.time()
    
        log.info('Tempo total de validacao'+ str(step_time_test_Total))
        log.info( 'Tempo total treinamento' + str((stepTimeTotalExecution)))
        log.info( 'Tempo total para gravar log' + str((totalTempoGravarArquivoLog)))
        log.info( 'Tempo total para gravar rede' + str((TotalTempoGravaRede)))
        log.info( 'Tempo total' + str((_end_time_total - _start_time_total)))
        batch_train = None 
        imgs = None
         
          
        return questaotipo0,questaotipo1,questaotipo2,questaotipo3,questaotipo4        
       

    def run_single_step(self, batch, config, step=None):
        _start_time = time.time()
        qtd = 100
        #batch_chunk = self.session.run(batch)
        treino=[]
        dataset = 0

        
        
        if (config.trainPosition >config.dataset_train.maxGrups -1):
            config.trainPosition = 0
        treino = config.dataset_train.batch[config.trainPosition]
        fetch = [self.global_step, self.model.accuracy, self.summary_op,
                         self.model.loss, self.check_op, self.optimizer,  self.model.all_preds, self.model.a]
      
          #treino= tf.convert_to_tensor(treino)
        try:
               if step is not None and (step % 100 == 0):
                 fetch += [self.plot_summary_op]
        except:
               pass
       
        fetch_values = self.session.run(
                  fetch, feed_dict=self.model.get_feed_dict2([treino[1],treino[2],treino[3],treino[4],treino[5],treino[6],fetch], step=step)
                )
      
        [step, accuracy, summary, loss] = fetch_values[:4]
       
           
        try:
                if self.plot_summary_op in fetch:
                    summary += fetch_values[-1]
        except:
                pass

        _end_time = time.time()
        return step, accuracy, summary, loss,  (_end_time - _start_time)

    def GravarCSV ( self,data_dict,fname):
      
       print("gravar arquivo: " + fname)
       os.makedirs(self.trainDir, exist_ok=True)
       fname = self.trainDir + "/" + fname +".csv"
       # Create file
       file =  open(fname, 'a',newline='')
       writer = csv.writer(file)   
       writer.writerow(data_dict) 
       
       file.close()
       
    def run_test(self,tipo,step ,dataset,is_train=False):
        _start_time = time.time()
        treino=[]
        accuracy_test = 0
        i =0
        while (i < dataset.maxGrups -1):
            treino = dataset.batch[i]
            [accuracy_teste_step, all_preds, all_targets]  = self.session.run(
                [self.model.accuracy, self.model.all_preds, self.model.a], feed_dict=self.model.get_feed_dict2([treino[1],treino[2],treino[3],treino[4],treino[5],treino[6]], is_training=False))
            accuracy_test =   accuracy_teste_step +accuracy_test
            self.add_batch( treino[0],treino[2],treino[3] ,all_preds, all_targets)
            i=i+1
        accuracy_test,avg_nr,avg_r = self.report(step,tipo)
        _end_time = time.time() 
        #tf.compat.v1.summary.scalar("loss/accuracy_test", (accuracy_test))
        self._ids=[]
        self._questions=[]
        self._answers=[]
        self._predictions=[]
        self._groundtruths=[]
        self.ArrarQuestoesErradas = []
        self.ArrayQuestoesCertas =[]
        questaotipo0 = self.ArrayTotalQuestoesCertas[0]/ ( self.ArrayTotalQuestoesErradas[0] +  self.ArrayTotalQuestoesCertas[0] )
        questaotipo1 = self.ArrayTotalQuestoesCertas[1]/ (self.ArrayTotalQuestoesErradas[1] +  self.ArrayTotalQuestoesCertas[1])
        questaotipo2 = self.ArrayTotalQuestoesCertas[2]/ (self.ArrayTotalQuestoesErradas[2] + self.ArrayTotalQuestoesCertas[2])
        questaotipo3 = self.ArrayTotalQuestoesCertas[3]/ (self.ArrayTotalQuestoesErradas[3] +  self.ArrayTotalQuestoesCertas[3])
        questaotipo4 = self.ArrayTotalQuestoesCertas[4]/ (self.ArrayTotalQuestoesErradas[4] +  self.ArrayTotalQuestoesCertas[4])
        totalQuestoes = self.ArrayTotalQuestoesErradas[0] +  self.ArrayTotalQuestoesCertas[0] + self.ArrayTotalQuestoesErradas[1] +  self.ArrayTotalQuestoesCertas[1] + self.ArrayTotalQuestoesErradas[2] + self.ArrayTotalQuestoesCertas[2] + self.ArrayTotalQuestoesErradas[3] +  self.ArrayTotalQuestoesCertas[3] + self.ArrayTotalQuestoesErradas[4] +  self.ArrayTotalQuestoesCertas[4] 
        totalQuestoesCertas = self.ArrayTotalQuestoesCertas[0] +   self.ArrayTotalQuestoesCertas[1] + self.ArrayTotalQuestoesCertas[2] +   self.ArrayTotalQuestoesCertas[3] +   self.ArrayTotalQuestoesCertas[4]
        
        self.ArrayTotalQuestoesCertas =[0,0,0,0,0] 
        self.ArrayTotalQuestoesErradas =[0,0,0,0,0]
        return (totalQuestoesCertas/totalQuestoes),(_end_time-_start_time),questaotipo0,questaotipo1,questaotipo2,questaotipo3,questaotipo4


    def report(self,step,tipo):

        #img, q, a = self.dataset.get_data(self._ids[0])
        #visualize_iqa( img, q, a)


        # report L2 loss
        log.info("Computing scores...")
        correct_prediction_nr = 0
        count_nr = 0
        correct_prediction_r = 0
        correctQuest =0
        errorQuest = 0
        count_r = 0
        
        
        for id,q,a, pred, gt in zip(self._ids,self._questions, self._answers ,self._predictions, self._groundtruths):
            for i in range(pred.shape[0]):
                # relational
       
                quest = question2str(q[i])
                #if int(id[i]) == 39950 : 
                #    print('oi')
                answer = answer2str(a[i])
                anserPred = np.zeros((len(a[i]))) 
                anserPred[np.argmax(pred[i,:])] = 1 
                anserPred1 = answer2str(anserPred)
                q_num = np.argmax(q[i][6:])
                if  q_num >= 2:
                    count_r += 1
                    
                    if np.argmax(pred[i, :]) == np.argmax(gt[i, :]):
                        self.ArrayTotalQuestoesCertas[q_num] +=1
                        correct_prediction_r += 1
                        correctQuest += 1 
                        #self.ArrayQuestoesCertas.append(str(int(id[i])))
                        self.ArrayQuestoesCertas.append("qestao id numero : " +str(int(id[i])))
                        self.ArrayQuestoesCertas.append(quest)
                        self.ArrayQuestoesCertas.append(answer)
                        self.ArrayQuestoesCertas.append(anserPred1)
                        self.ArrayQuestoesCertas.append("qestao do tipo : " +str(q_num)) 
                        self.ArrayQuestoesCertas.append("tipo : Relacional")
                    else:
                        errorQuest += 1 
                        self.ArrayTotalQuestoesErradas[q_num] += 1
                        #self.ArrarQuestoesErradas.append(str(int(id[i])))
                        self.ArrarQuestoesErradas.append ("qestao id numero : " + str(int(id[i])))
                        self.ArrarQuestoesErradas.append(quest)
                        self.ArrarQuestoesErradas.append("Reposta errada:" + anserPred1)  
                        self.ArrarQuestoesErradas.append("Resposta certa:" + answer) 
                        self.ArrarQuestoesErradas.append("qestao do tipo : " +str(q_num))
                        self.ArrarQuestoesErradas.append("tipo : Relacional")

                # non-relational
                else:
                    count_nr += 1
                    if np.argmax(pred[i, :]) == np.argmax(gt[i, :]):
                        self.ArrayTotalQuestoesCertas[q_num] +=1
                        correctQuest += 1 
                        correct_prediction_nr += 1
                        #self.ArrayQuestoesCertas.append(str(int(id[i])))
                        self.ArrayQuestoesCertas.append("qestao id numero : " +str(int(id[i])))
                        self.ArrayQuestoesCertas.append(quest)
                        self.ArrayQuestoesCertas.append(answer)
                        self.ArrayQuestoesCertas.append(anserPred1)
                        self.ArrayQuestoesCertas.append("qestao do tipo : " +str(q_num)) 
                        self.ArrayQuestoesCertas.append("tipo : Nao-Relacional")
                    else:
                        errorQuest +=1
                        self.ArrayTotalQuestoesErradas[q_num] += 1
                        #self.ArrarQuestoesErradas.append ( str(int(id[i])))
                        self.ArrarQuestoesErradas.append ("qestao id numero : " + str(int(id[i])))
                        self.ArrarQuestoesErradas.append(quest)
                        self.ArrarQuestoesErradas.append("Reposta errada:" + anserPred1)  
                        self.ArrarQuestoesErradas.append("Resposta certa:" + answer) 
                        self.ArrarQuestoesErradas.append("qestao do tipo : " +str(q_num)) 
                        self.ArrarQuestoesErradas.append("tipo : Nao-Relacional")

        avg_nr = float(correct_prediction_nr)/count_nr
        log.info("Average accuracy of non-relational questions: {}%".format(avg_nr*100))
        avg_r = float(correct_prediction_r)/count_r
        log.info("Average accuracy of relational questions: {}%".format(avg_r*100))
        avg = float(correct_prediction_r+correct_prediction_nr)/(count_r+count_nr)
        log.info("Average accuracy: {}%".format(avg*100))
      
        return avg,avg_nr,avg_r
        #self.GravarArquivo(errorQuest,self.ArrarQuestoesErradas,"questaoErrada" +tipo ,file_folder,step )
        #self.GravarArquivo(correctQuest,self.ArrayQuestoesCertas,"questaoCerta"+ tipo,file_folder,step)
      
       

    def GravarArquivo (self,qtd, data_dict,fname, file_folder,step):
      fname = fname +"_" +str(qtd) + '.json'
      print("gravar arquivo: " + fname + " qtd: " +  str(len(data_dict)))
      os.makedirs(file_folder+'//'+'processamento'+str(step), exist_ok=True)
      fname = file_folder+'processamento'+str(step) + "/" + fname
      # Create file
      with open(fname, 'w') as outfile:
        json.dump(data_dict, outfile, ensure_ascii=False, indent=4) 
        outfile.close()
    
    def log_step_message(self, step, accuracy, accuracy_test, loss, step_time, step_time_training,is_train=True):
        self.acuracy.append(accuracy)
        self.step.append(step)
        if step_time == 0:
            step_time = 0.001
        log_fn = (is_train and log.info or log.infov)
        log_fn((" [{split_mode:5s} step {step:4d}] " +
                "Loss: {loss:.5f} " +
                "Accuracy : {accuracy:.2f} "
                "Accuracy test: {accuracy_test:.2f} " + 
                 "({sec_per_batch:.3f} sec/batch, {instance_per_sec:.3f} instances/sec) " +
                 "({sec_per_batch_test:.3f} sec/batch teste)"
                ).format(split_mode=(is_train and 'train' or 'val'),
                         step=step,
                         loss=loss,
                         accuracy=accuracy*100,
                         accuracy_test=accuracy_test*100,
                         sec_per_batch=step_time,
                         instance_per_sec=self.batch_size / step_time , 
                         sec_per_batch_test = step_time_training
                         )
               )

    
    def Clear(self): 
      del self.model
      self.model = None  
    