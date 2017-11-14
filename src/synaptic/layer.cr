module Synaptic
  class Layer
    enum Connection_Type
      ALL_TO_ALL,
      ONE_TO_ONE,
      ALL_TO_ELSE
    end

    enum Gate_Type
      INPUT,
      OUTPUT,
      ONE_TO_ONE
    end

    getter :neurons

    def initialize(@size : Int32)
      @neurons = [] of Neuron
      @connected_to = [] of Neuron

      # Create our neurons
      @size.times { @neurons << Neuron.new }
    end

    def activate!(input : Array(Float64)? = nil) : Array(Float64)
      activations = [] of Float64

      if input
        if input.size != @size
          raise InputSizeMismatchError.new
        end

        @neurons.each_with_index do |index, neuron|
          activation = neuron.activate!(input[index])
          activations.push activation
        end
      else
        @neurons.each do |neuron|
          activation = neuron.activate!
          activations.push(activation)
        end
      end

      activations
    end

    # Propagates the error on all the neurons of the layer
    def propagate!(learning_rate : Float64, target : Array(Float64)? = nil)
      if target
        if target.size != @size
          raise TargetSizeMismatchError.new
        end

        @neurons.each_with_index do |index, neuron|
          neuron.propagate!(learning_rate, target[index])
        end
      else
        @neurons.each do |neuron|
          neuron.propagate!(learning_rate)
        end
      end
    end

    # Projects a connection from this layer to another one
    def project(target_layer, kind, weights)
      if target_layer.is_a?(Network)
        target_layer = target_layer.layers.input
      end

      if !connected_to?(target_layer)
        return LayerConnection.new(self, target_layer, kind, weights)
      end
    end
  end

  class InputSizeMismatchError < Exception
    def initialize(message = "Input size and Layer size must be the same to activate")
      super(message)
    end
  end

  class TargetSizeMismatchError < Exception
    def initialize(message = "Target size and Layer size must be the same to propagate")
      super(message)
    end
  end
end
