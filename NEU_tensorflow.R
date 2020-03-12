library("tensorflow");library("keras")
#--------------------------------------#
d<-2; x<-k_constant(c(1,2)) # TEST
# Diagram Representation
#---------------------------#
# x   -> x-c              -> 
  #         -> 1/σ^2 -> norm -> I(0<<1) -> ψ -> e^ -> X (affine) ->x -> W





# Shift only layer (Trainable)      # TEMP
#-------------------------------#
shift_layer = layer_dense(units=d,
                      activation = "linear",
                      trainable=TRUE,
                      input_shape = d,
                      bias_initializer = "random_normal") 



# Duplicatoin Layer x -> (x,x) (Untrainable)
#-------------------------------------------#
duplicate <- R6::R6Class("KerasLayer",
                                  
                                  inherit = KerasLayer,
                                  
                                  public = list(
                                    
                                    call = function(x, mask = NULL) {
                                      k_concatenate(c(x, x), axis = 2)
                                    },
                                    
                                    compute_output_shape = function(input_shape) {
                                      input_shape[[2]] <- input_shape[[2]] * 2L 
                                      input_shape
                                    }
                                  )
)

# Create layer wrapper function
layer_duplicate <- function(object) {
  create_layer(duplicate, object)
}




# Rescale Layer ()
#---------------------------------#
rescale_layer = layer_dense(units=d,
                      activation = "linear",
                      trainable=TRUE,
                      input_shape = d,
                      bias_initializer = "random_normal") 


# Bump Layer
#---------------------------------#
componentwise_bump_function <- R6::R6Class("KerasLayer",
                         
                         inherit = KerasLayer,
                         
                         public = list(
                           
                           call = function(x, mask = NULL) {
                             # Initializes Norm
                             norm_x = tf$math$abs(x)
                             # Determines If Apply Threshold Or Not
                             cond = tf$less(norm_x,1)
                             # Outputs Value of bump function
                             out = tf$where(cond, tf$math$exp(-1/norm_x), k_constant(0))
                           })
)

# Create layer wrapper function
layer_componentwise_bump_function <- function(object) {
  create_layer(componentwise_bump_function, object)
}



# Fully Trainable 
#---------------------------------#
R_layer = layer_dense(units=d,
            activation = "linear",
            trainable=TRUE,
            input_shape = d,
            bias_initializer = "random_normal") 





# Multiply Outputs
#---------------------------------#



# Create Custom "Super" Layer
#----------------------------#
# x   -> x-c              -> 
#         -> 1/σ^2 -> norm -> I(0<<1) -> ψ -> e^ -> X (affine) ->x -> W


