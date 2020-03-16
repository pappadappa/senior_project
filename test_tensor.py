# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 15:04:08 2020

@author: Admin
"""
import tensorflow as tf


#Rank 0

# mammal = tf.Variable("Elephant", tf.string)
# ignition = tf.Variable(451, tf.int16)
# floating = tf.Variable(3.14159265359, tf.float64)
# its_complicated = tf.Variable(12.3 - 4.85j, tf.complex64)

# print(mammal)

# Rank 1

# mystr = tf.Variable(["Hello"], tf.string)
# cool_numbers  = tf.Variable([3.14159, 2.71828], tf.float32)
# first_primes = tf.Variable([2, 3, 5, 7, 11], tf.int32)
# its_very_complicated = tf.Variable([12.3 - 4.85j, 7.5 - 6.23j], tf.complex64)

# print(first_primes)

# Rank 2

# mymat = tf.Variable([[7],[11]], tf.int16)
# myxor = tf.Variable([[False, True],[True, False]], tf.bool)
# linear_squares = tf.Variable([[4], [9], [16], [25]], tf.int32)
# squarish_squares = tf.Variable([ [4, 9], [16, 25] ], tf.int32)
# rank_of_squares = tf.rank(squarish_squares)
# mymatC = tf.Variable([[7],[11]], tf.int32)

# High rank

# my_image = tf.zeros([10, 299, 299, 3])

#Shapez

# zeros = tf.zeros(my_matrix.shape[1])

#Changing the shape of a tf.Tensor

rank_three_tensor = tf.ones([3, 4, 5])
matrix = tf.reshape(rank_three_tensor, [6, 10])  # Reshape existing content into
                                                 # a 6x10 matrix
matrixB = tf.reshape(matrix, [3, -1])  #  Reshape existing content into a 3x20
                                       # matrix. -1 tells reshape to calculate
                                       # the size of this dimension.
matrixAlt = tf.reshape(matrixB, [4, 3, -1])  # Reshape existing content into a
                                             #4x3x5 tensor



