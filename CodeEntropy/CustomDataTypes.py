import numpy as nmp
import sys
from collections import deque
from CodeEntropy.FunctionCollection import Utils

class LinkedListNode(object):
	def __init__(self, arg_value):
		self.value = arg_value
		self.nextNode = None

	def __eq__(self, arg_otherNode):
		return self.value == arg_otherNode.value

class DoublyLinkedListNode(LinkedListNode):
	def __init__(self, arg_value):
		super().__init__(arg_value)
		self.previousNode = None

	def __iter__(self):
		return self

class CircularLinkedList(object):
	""" 
	A linked list where each node is connected to its next node 
	and the last node is linked with the first node,
	thus completing a circle"""

	def __init__(self):
		self.currentPosition = None
		self.startPosition = None
		self.count = 0

	def append(self, arg_node):
		if self.count == 0:
			self.startPosition = arg_node
			self.currentPosition = arg_node
			arg_node.nextNode = self.currentPosition
			self.incr_count()

		else:
			self.currentPosition.nextNode = arg_node
			self.currentPosition = arg_node
			arg_node.nextNode = self.startPosition
			self.incr_count()

		return

	def incr_count(self):
		self.count += 1
		return

	def reset_position(self):
		self.currentPosition = self.startPosition
		return

	def __len__(self):
		return self.count

	def get_current_position(self):
		return self.currentPosition

	def move_right_of_current_position(self):
		self.currentPosition = self.currentPosition.nextNode
		return
#END

class DoublyLinkedList(object):
	""" Each node is connected to its next and previous node."""
	def __init__(self):
		self.startPosition = None
		self.endPosition = None
		self.currentPosition = None
		self.count = 0


	def append(self, arg_doubleNode):
		if self.count == 0:
			# means it points to nothing
			# initialize it now
			self.startPosition = arg_doubleNode
			self.endPosition = arg_doubleNode
			self.currentPosition = arg_doubleNode
			self.incr_count()

		else:
			# add it to the right of the currently last position
			self.endPosition.nextNode = arg_doubleNode
			self.endPosition = arg_doubleNode
			self.currentPosition = self.endPosition
			self.incr_count()

		return

	def incr_count(self):
		self.count += 1
		return

	def decr_count(self):
		self.count -= 1
		assert(self.count >= 0)
		return

	def __len__(self):
		return self.count

	def get_current_position(self):
		return self.currentPosition

	def get_node_at(self, arg_index):
		#0-indexed
		try:
			assert(arg_index < self.count and arg_index >= 0)
		except:
			raise IndexError('{} exceeds the last index of the list ({})'.format(arg_index, self.count - 1))
		
		# backup the current postion
		backupCurrentPosition = self.get_current_position()

		idx = 0
		self.move_to_start()
		idxNode = self.get_current_position()
		while idx != arg_index:
			idxNode = idxNode.nextNode
			idx += 1

		self.currentPosition = backupCurrentPosition
		return idxNode


	def move_to_start(self):
		self.currentPosition = self.startPosition
		return 1

	def move_to_end(self):
		self.currentPosition = self.endPosition
		return 1

	def move_right_of_current_position(self):
		self.currentPosition = self.currentPosition.nextNode
		return 

	def move_left_of_current_position(self):
		self.currentPosition = self.currentPosition.previousNode
		return 

	def remove_node_and_move_right(self):
		""" Remove the node pointed by the currentPosition and then move to its nextNode """
		leftNode = self.currentPosition.previousNode
		rightNode = self.currentPosition.nextNode

		if leftNode != None:
			leftNode.nextNode = rightNode
		if rightNode != None:
			rightNode.previousNode = leftNode

		self.currentPosition = rightNode
		self.decr_count()
		return 1

	def remove_node_and_move_left(self):
		""" Remove the node pointed by the currentPosition and then move to its previousNode """
		leftNode = self.currentPosition.previousNode
		rightNode = self.currentPosition.nextNode

		if leftNode != None:
			leftNode.nextNode = rightNode
		if rightNode != None:
			rightNode.previousNode = leftNode

		self.currentPosition = previousNode
		self.decr_count()
		return 1

	def clear_all(self):
		""" Basically, point all its memebr pointers to None and reset its count to 0 """
		self.startPosition = None
		self.endPosition = None
		self.currentPosition = None
		self.count = 0
		return


	def iterator(self):
		assert(self.count != 0)

		self.move_to_start()
		while self.currentPosition:
			yield self.currentPosition
			self.move_right_of_current_position()
#END

class CircularDoublyLinkedList(DoublyLinkedList):
	""" 
	Each node is connected to its next and previous node. At
	the end of the complete circle, the next node to the last 
	node is the first node. This is a derived class of the 
	`DoublyLinkedList` class.
	"""

	def __init__(self):
		super().__init__()


	def append(self, arg_doubleNode):
		if self.count == 0:
			# means it points to nothing
			# initialize it now
			self.startPosition = arg_doubleNode
			self.endPosition = arg_doubleNode
			self.currentPosition = arg_doubleNode
			self.incr_count()

		else:
			# add it to the right of the currently last position
			self.endPosition.nextNode = arg_doubleNode
			self.startPosition.previousNode = arg_doubleNode

			arg_doubleNode.previousNode = self.endPosition
			arg_doubleNode.nextNode = self.startPosition

			self.endPosition = arg_doubleNode
			self.currentPosition = self.endPosition

			self.incr_count()

		return

	def incr_count(self):
		self.count += 1
		return

	def decr_count(self):
		self.count -= 1
		assert(self.count >= 0)
		return

	def __len__(self):
		return self.count

	def get_current_position(self):
		return self.currentPosition

	def get_node_at(self, arg_index):
		#0-indexed
		try:
			assert(arg_index < self.count and arg_index >= 0)
		except:
			raise IndexError('{} exceeds the last index of the list ({})'.format(arg_index, self.count - 1))
		
		# backup the current postion
		backupCurrentPosition = self.get_current_position()

		idx = 0
		self.move_to_start()
		idxNode = self.get_current_position()
		while idx != arg_index:
			idxNode = idxNode.nextNode
			idx += 1

		self.currentPosition = backupCurrentPosition
		return idxNode


	def move_to_start(self):
		self.currentPosition = self.startPosition
		return 1

	def move_to_end(self):
		self.currentPosition = self.endPosition
		return 1

	def move_right_of_current_position(self):
		self.currentPosition = self.currentPosition.nextNode
		return 

	def move_left_of_current_position(self):
		self.currentPosition = self.currentPosition.previousNode
		return 

	def remove_node_and_move_right(self):
		""" Remove the node pointed by the currentPosition and then move to its nextNode """
		leftNode = self.currentPosition.previousNode
		rightNode = self.currentPosition.nextNode

		if leftNode != None:
			leftNode.nextNode = rightNode
		if rightNode != None:
			rightNode.previousNode = leftNode

		self.currentPosition = rightNode
		self.decr_count()
		return 1

	def remove_node_and_move_left(self):
		""" Remove the node pointed by the currentPosition and then move to its previousNode """
		leftNode = self.currentPosition.previousNode
		rightNode = self.currentPosition.nextNode

		if leftNode != None:
			leftNode.nextNode = rightNode
		if rightNode != None:
			rightNode.previousNode = leftNode

		self.currentPosition = previousNode
		self.decr_count()
		return 1

	def clear_all(self):
		""" Basically, point all its memebr pointers to None and reset its count to 0 """
		self.startPosition = None
		self.endPosition = None
		self.currentPosition = None
		self.count = 0
		return


	def iterator_cw(self):
		"""
		Iterate through next nodes (clockwise) from the start to the end
		of the circular doubly linked list.
		"""
		# only iterate if the list is non-empty
		assert(self.count != 0)

		# create a backup of the current poisition to restore it later
		backupCurrentPosition = self.currentPosition

		# begin iteration
		self.move_to_start()
		while self.currentPosition:
			yield self.currentPosition
			if self.currentPosition == self.endPosition:
				break
			else:
				self.move_right_of_current_position()

		# restore the original current position after iteration is completed
		self.currentPosition = backupCurrentPosition

	def iterator_ccw(self):
		"""
		Iterate through previous nodes (counter-clockwise) from the start to the end
		of the circular doubly linked list.
		"""
		# only iterate if the list is non-empty
		assert(self.count != 0)

		# create a backup of the current poisition to restore it later
		backupCurrentPosition = self.currentPosition

		# begin iteration
		self.move_to_end()
		while self.currentPosition:
			yield self.currentPosition
			if self.currentPosition == self.startPosition:
				break
			else:
				self.move_left_of_current_position()

		# restore the original current position after iteration is completed
		self.currentPosition = backupCurrentPosition

#END

class TreeNode(object):
	""" A generic data structure that depicts a node for a binary tree. 
	Has a pointer to one parent and 2 children """

	def __init__(self, arg_leftChildNode, arg_rightChildNode, arg_value):
		self.leftChildNode = arg_leftChildNode
		self.rightChildNode = arg_rightChildNode
		self.value = arg_value

	def __lt__(self, arg_otherNode):
		assert(self.value != None and arg_otherNode.value != None)
		return self.value < arg_otherNode.value

	def __gt__(self, arg_otherNode):
		assert(self.value != None and arg_otherNode.value != None)
		return self.value > arg_otherNode.value


#END

class BinaryTree(object):
	""" A rooted bindary tree which will be defined by a root tree node. It will be populated by links 
	to the root nodes and their subsequent links"""
	def __init__(self):
		self.rootNode = None
		self.nodeCount = 0

	def __len__(self):
		return self.nodeCount
	#END


	def add_node(self, arg_node):
		if self.rootNode == None:
			# arg_node is the root, add it likewise
			self.rootNode = arg_node
			self.incr_nodeCount()
			return

		else:
			self.add_node_to_rooted_tree(arg_parentNode = self.rootNode, arg_node = arg_node)
			return
	#END

	def add_node_to_rooted_tree(self, arg_parentNode, arg_node):
		if arg_node < arg_parentNode:
			if arg_parentNode.leftChildNode == None:
				arg_parentNode.leftChildNode = arg_node
				arg_node.parentNode = arg_parentNode
				self.incr_nodeCount()
				return
			else:
				self.add_node_to_rooted_tree(arg_parentNode = arg_parentNode.leftChildNode, arg_node = arg_node)

		elif arg_node > arg_parentNode:
			if arg_parentNode.rightChildNode == None:
				arg_parentNode.rightChildNode =arg_node
				arg_node.parentNode = arg_parentNode
				self.incr_nodeCount()
				return
			else:
				self.add_node_to_rooted_tree(arg_parentNode = arg_parentNode.rightChildNode, arg_node = arg_node)

	def list_in_order(self):
		outList = []
		try:
			assert(self.rootNode != None)
		except:
			# print('Tree is non-existent. It will return an empty list')
			return outList

		self.generate_IN_order_list_from_node(arg_node = self.rootNode, arg_outList = outList)
		return outList

	#END	



	def generate_IN_order_list_from_node(self, arg_node, arg_outList):
		if arg_node.leftChildNode == None:
			pass
		else:
			self.generate_IN_order_list_from_node(arg_node = arg_node.leftChildNode, arg_outList = arg_outList)

		arg_outList.append(arg_node.value) 

		if arg_node.rightChildNode == None:
			pass
		else:
			self.generate_IN_order_list_from_node(arg_node = arg_node.rightChildNode, arg_outList = arg_outList) 

		return arg_outList
	#END


	
	def incr_nodeCount(self):
		# Utils.printflush('Increasing node count from {}'.format(self.nodeCount), end = '')
		self.nodeCount += 1
		# Utils.printflush(' to {}'.format(self.nodeCount))
		return

	#END

	def decr_nodeCount(self):
		# Utils.printflush('Decreasing node count from {}'.format(self.nodeCount), end = '')
		self.nodeCount -= 1
		# Utils.printflush(' to {}'.format(self.nodeCount))
		return
	#END

#END

class AxesSystem(object):
	"""
	A class that contains the orthogonal axes and 
	the origin of a N-dimensional space.
	"""

	def __init__(self, arg_ndim):
		self.basisVectors = nmp.zeros((arg_ndim, arg_ndim))
		self.origin = nmp.zeros(arg_ndim)

	def __str__(self):
		retStr = "Basis vectors:\n"
		for i in range(self.get_dimension()):
			retStr = "{}|".format(retStr)
			for j in range(self.get_dimension()):
				retStr = "{}{:>8.3f}".format(retStr, self.basisVectors[i][j])
			retStr = "{}|\n".format(retStr)

		retStr = "{}\n\nOrigin:\n"
		for k in range(self.get_dimension()):
			retStr = "{}{:>8.3f}".format(retStr, self.origin[k])
		return retStr

	def get_basis_vectors(self):
		return self.basisVectors

	def get_dimension(self):
		""" Return the dimensionality of the axes system."""
		return len(self.origin)

	def transpose(self):
		""" Return a matrix where the columns are system's basis vectors.""" 
		return nmp.tramspose(self.basisVectors)







