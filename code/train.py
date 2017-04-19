import numpy as np 
import tensorflow as tf 
from DataSet import dataSet
import config
import cane
import random



#load data
graph_path='graph.txt'
text_path='data.txt'

data=dataSet(text_path,graph_path)

# start session

with tf.Graph().as_default():
	sess=tf.Session()
	with sess.as_default():
		model=cane.Model(data.num_vocab,data.num_nodes)
		opt=tf.train.AdamOptimizer(config.lr)
		train_op=opt.minimize(model.loss)
		sess.run(tf.global_variables_initializer())

		#training
		print 'start training.......'

		for epoch in range(config.num_epoch):
			loss_epoch=0
			batches=data.generate_batches()
			h1=0
			num_batch=len(batches)
			for i in range(num_batch):
				batch=batches[i]

				node1,node2,node3=zip(*batch)
				node1,node2,node3=np.array(node1),np.array(node2),np.array(node3)
				text1,text2,text3=data.text[node1],data.text[node2],data.text[node3]

				feed_dict={
					model.Text_a:text1,
					model.Text_b:text2,
					model.Text_neg:text3,
					model.Node_a:node1,
					model.Node_b:node2,
					model.Node_neg:node3
				}

				# run the graph 
				_,loss_batch = sess.run([train_op,model.loss],feed_dict=feed_dict)

				loss_epoch += loss_batch
			print 'epoch: ',epoch+1,' loss: ',loss_epoch

		file=open('embed.txt','wb')
		batches=data.generate_batches(mode='add')
		num_batch=len(batches)
		embed=[[] for _ in range(data.num_nodes)]
		for i in range(num_batch):
			batch=batches[i]

			node1,node2,node3=zip(*batch)
			node1,node2,node3=np.array(node1),np.array(node2),np.array(node3)
			text1,text2,text3=data.text[node1],data.text[node2],data.text[node3]

			feed_dict={
				model.Text_a:text1,
				model.Text_b:text2,
				model.Text_neg:text3,
				model.Node_a:node1,
				model.Node_b:node2,
				model.Node_neg:node3
			}

			# run the graph 
			convA,convB,TA,TB = sess.run([model.convA,model.convB,model.N_A,model.N_B],feed_dict=feed_dict)
			for i in range(config.batch_size):
				em=list(convA[i])+list(TA[i])
				embed[node1[i]].append(em)
				em=list(convB[i])+list(TB[i])
				embed[node2[i]].append(em)
		for i in range(data.num_nodes):
			if embed[i]:
				# print embed[i]
				tmp=np.sum(embed[i],axis=0)/len(embed[i])
				file.write(' '.join(map(str,tmp))+'\n')
			else:
				file.write('\n')
