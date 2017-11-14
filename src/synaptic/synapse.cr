module Synaptic
  class Synapse
    @@connections = 0

    class_getter :connections
    getter :from, :to, :id
    property :gain, :weight, :gater

    @id : Int32 = Synapse.uid
    @gain : Float64 = 1.0
    @gater : Neuron? = nil
    @weight : Float64 = Random.rand(-1.0...1.0)

    def initialize(@from : Neuron, @to : Neuron, weight : Float64? = nil)
      @weight = weight unless weight.nil?
    end

    def self.uid
      @@connections += 1
    end
  end
end
