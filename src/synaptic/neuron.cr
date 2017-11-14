module Synaptic
  class Neuron
    alias SquashFunction = Proc((Float64), Bool, Float64)

    @@instances = 0

    SQUASH = {
      logistic: ->(x : Float64, derivate : Bool) {
        fx = 1.0 / (1.0 + Math.exp(-x))
        if (!derivate)
          return fx
        end
        fx * (1.0 - fx)
      },
      tanh: ->(x : Float64, derivate : Bool) {
        if (derivate)
          return 1.0 - (Math.tanh(x) ** 2.0)
        end
        Math.tanh(x)
      },
      identity: ->(x : Float64, derivate : Bool) {
        derivate ? 1.0 : x
      },
      hlim: ->(x : Float64, derivate : Bool) {
        derivate ? 1.0 : x > 0.0 ? 1.0 : 0.0
      },
      relu: ->(x : Float64, derivate : Bool) {
        if derivate
          return x > 0.0 ? 1.0 : 0.0
        end
        x > 0.0 ? x : 0.0
      },
    }

    getter :id, :state, :old, :activation, :self_connection,
      :neighbors, :connections, :error, :trace

    @id : Int32 = Neuron.uid
    @state : Float64 = 0.0
    @old : Float64 = 0.0
    @influence : Float64 = 0.0
    @activation : Float64 = 0.0
    @derivative : Float64 = 0.0
    @neighbors : Hash(Int32, Neuron) = {} of Int32 => Neuron
    @squash : SquashFunction = SQUASH[:logistic]
    @bias : Float64 = Random.rand(-1.0...1.0)

    def initialize
      @connections = {
        inputs:    {} of Int32 => Synapse,
        projected: {} of Int32 => Synapse,
        gated:     {} of Int32 => Synapse,
      }

      @error = {
        :responsibility => 0.0,
        :projected      => 0.0,
        :gated          => 0.0,
      }

      @trace = {
        eligability: {} of Int32 => Float64,
        extended:    {} of Int32 => Array(Float64),
        influences:  {} of Int32 => Array(Synapse),
      }

      @self_connection = Synapse.new(self, self, 0.0) # weight = 0.0 -> not connected
    end

    def activate!(input : Float64? = nil)
      if input
        @activation = input
        @derivative = 0.0
        @bias = 0.0
        return @activation
      end

      @old = @state.dup
      # TODO: Figure out why I need a `not_nil!` here
      @state = @self_connection.not_nil!.gain * @self_connection.not_nil!.weight * @state + @bias

      @connections[:inputs].each do |key, input|
        @state += input.from.activation * input.weight * input.gain
      end

      @activation = @squash.call(@state, false)
      @derivative = @squash.call(@state, true)

      # Update traces
      influences = {} of Int32 => Float64
      @trace[:extended].each do |key, value|
        # Extended eligability trace
        neuron = @neighbors[key]

        # If gated neuron's self_connection is gated by this unit, the influence
        # keeps track of the neuron's old state
        influence = @self_connection.not_nil!.gater == self ? neuron.old : 0

        # Index runs over all the incoming connections to the gated neuron that
        # are gated by this unit
        @trace[:influences][neuron.id].each do |incoming|
          # Captures the effect that has an input connection to this unit, on a
          # neuron that is gated by this unit
          influence += incoming.weight * incoming.from.activation
        end

        influences[neuron.id] = influence.to_f
      end

      @connections[:inputs].each do |key, input|
        # Eligability trace
        @trace[:eligability][input.id] = @self_connection.not_nil!.gain *
                                         @self_connection.not_nil!.weight *
                                         @trace[:eligability][input.id] + input.gain *
                                                                          input.from.activation

        @trace[:extended].each do |key, value|
          # Extended eligability trace
          xtrace = value
          neuron = @neighbors[key]
          influence = influences[neuron.id]
        end
      end

      # Update gated connection's gains
      @connections[:gated].each do |key, connection|
        connection.gain = @activation
      end

      @activation
    end

    # Back propogate the error
    def propagate!(learning_rate : Float64? = 0.1, target_value : Float64? = nil) : Float64
      # Error accumulator
      error = 0.0

      # Whether or not this neuron is in the output layer
      is_output? = !target_value.nil?

      if is_output?
        @error[:responsibility] = @error[:projected] = target_value.not_nil! - @activation
      else
        # The rest of the neurons compute their error responsibilities by backpropagation
        @connections[:projected].each do |key, connection|
          neuron = connection.to
          error += neuron.error[:responsibility] * connection.gain * connection.weight
        end

        # Projected error responsibility
        @error[:projected] = @derivative * error

        # Reset the accumulator
        error = 0.0

        # Error responsibilities from all the connections gated by this neuron
        @trace[:extended].each do |id, _|
          neuron = @neighbors[id] # Gated neuron
          # If gated neuron's selfconnection is gated by this neuron
          influence = neuron.self_connection.not_nil!.gater == self ? neuron.old : 0.0

          # Index runs over all the connections to the gated neuron that are gated by this neuron
          @trace[:influences][id].each do |input|
            influence += input.weight * @trace[:influences][neuron.id][id].from.activation
          end

          error += neuron.error[:responsibility] * influence
        end

        # Gated error responsibility
        @error[:gated] = @derivative * error

        # Error responsibility
        @error[:responsibility] = @error[:projected] + @error[:gated]
      end

      @connections[:inputs].each do |id, input|
        gradient = @error[:projected] * @trace[:eligability][input.id]
        @trace[:extended].each do |neuron_id, _|
          neuron = @neighbors[neuron_id]
          gradient += neuron.error[:responsibility] * @trace[:extended][neuron_id][input.id]
        end
        input.weight += learning_rate * gradient
      end

      # Adjust bias
      @bias += learning_rate * @error[:responsibility]
    end

    def project!(neuron : Neuron, weight : Float64? = nil) : Synapse
      # Self connection?
      if neuron == self
        @self_connection.not_nil!.weight = 1.0
        return @self_connection.not_nil!
      end

      # Save this in case we need it later
      connection : Synapse? = nil

      # Check if connection already exists
      connected = connected_to?(neuron)
      if (connected && !connected.is_a?(Bool) && connected[:kind] == :projected)
        # Update connection
        if weight
          connected[:connection].weight = weight
        end
        # Return existing connection
        return connected[:connection]
      else
        connection = Synapse.new(self, neuron, weight)
      end

      @connections[:projected][connection.id] = connection
      @neighbors[neuron.id] = neuron
      neuron.connections[:inputs][connection.id] = connection
      neuron.trace[:eligability][connection.id] = 0.0

      neuron.trace[:extended].each do |_, trace|
        trace[connection.id] = 0.0
      end

      connection
    end

    def gate!(connection : Synapse)
      # Add connection to gated list
      @connections[:gated][connection.id] = connection

      neuron = connection.to
      if !(@trace[:extended].has_key(neuron.id))
        # Extended trace
        @neighbors[neuron.id] = neuron
        xtrace = @trace[:extended][neuron.id] = {} of Int32 => Array(Float64)
        @connections[:inputs].each do |_, input|
          xtrace[input.id] = 0.0
        end
      end

      # Keep track
      if (@trace[:influences].has_key(neuron.id))
        @trace[:influences][neuron.id].push(connection)
      else
        @trace[:influences][neuron.id] = [connection]
      end

      # Set gater
      connection.gater = self
    end

    def self_connected? : Bool
      @self_connection.not_nil!.weight != 0.0
    end

    def connected_to?(neuron : Neuron) : Bool | NamedTuple(kind: Symbol, connection: Synapse)
      if self == neuron
        if self_connected?
          return {kind: :self_connection, connection: @self_connection.not_nil!}
        else
          return false
        end
      end

      @connections.keys.each do |kind|
        @connections[kind].each do |_, connection|
          return {kind: kind, connection: connection}
        end
      end

      false
    end

    # Clears all the traces (the neuron forgets it's context, but the connections remain intact)
    def clear!
      @trace[:eligability].each do |_, trace|
        trace = 0.0
      end

      @trace[:extended].each do |key, _|
        @trace[:extended][key].each do |extended|
          extended = 0.0
        end
      end

      @error[:responsibility] = @error[:projected] = @error[:gated] = 0.0
    end

    # All the connections are randomized and the traces are cleared
    def reset!
      clear!

      @connections.each do |kind, _|
        @connections[kind].each do |key, connection|
          @connections[kind][key].weight = Random.rand(-1.0...1.0)
        end
      end

      @bias = Random.rand(-1.0...1.0)
      @old = @state = @activation = @influence = @activation = @derivative = 0.0
    end

    # Hardcodes the behaviour of the neuron into an optimized function
    def optimize
    end

    def self.uid : Int32
      @@instances += 1
    end

    def self.quantity
      {neurons: @@instances, connections: Synapse.connections}
    end
  end
end
