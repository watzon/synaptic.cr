require "./synaptic/*"

module Synaptic
  # TODO Put your code here
end

input = Synaptic::Layer.new(2)
hidden = Synaptic::Layer.new(3)
output = Synaptic::Layer.new(1)

input.activate!
input.propagate!(0.2)
# hidden.activate!
# output.activate!

pp input
