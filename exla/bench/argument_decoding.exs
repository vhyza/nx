# --- BEFORE ---
# Operating System: macOS
# CPU Information: Apple M2
# Number of Available Cores: 8
# Available memory: 16 GB
# Elixir 1.20.4
# Erlang 29.1.2

# Benchmark suite executing with the following configuration:
# warmup: 1 s
# time: 3 s
# memory time: 0 ns
# reduction time: 0 ns
# parallel: 8
# inputs: none specified
# Estimated total run time: 8 s

# Benchmarking device inputs ...
# Benchmarking host inputs ...

# Name                    ips        average  deviation         median         99th %
# device inputs        2.85 K        0.35 ms     ±6.85%        0.35 ms        0.43 ms
# host inputs          0.87 K        1.15 ms     ±3.74%        1.15 ms        1.32 ms

# Comparison:
# device inputs        2.85 K
# host inputs          0.87 K - 3.29x slower +0.80 ms
# ---
# Name                    ips        average  deviation         median         99th %
# device inputs        2.82 K        0.36 ms     ±7.30%        0.35 ms        0.44 ms
# host inputs          0.87 K        1.15 ms     ±2.42%        1.15 ms        1.22 ms

# Comparison:
# device inputs        2.82 K
# host inputs          0.87 K - 3.24x slower +0.80 ms
# ---
# Name                    ips        average  deviation         median         99th %
# device inputs        2.87 K        0.35 ms     ±7.48%        0.35 ms        0.46 ms
# host inputs          0.85 K        1.18 ms     ±4.97%        1.17 ms        1.54 ms

# Comparison:
# device inputs        2.87 K
# host inputs          0.85 K - 3.38x slower +0.83 ms
# ---
# Name                    ips        average  deviation         median         99th %
# device inputs        2.91 K        0.34 ms     ±7.28%        0.34 ms        0.43 ms
# host inputs          0.86 K        1.16 ms     ±3.14%        1.16 ms        1.22 ms

# Comparison:
# device inputs        2.91 K
# host inputs          0.86 K - 3.37x slower +0.81 ms
# ---
# Name                    ips        average  deviation         median         99th %
# device inputs        2.74 K        0.36 ms    ±33.45%        0.35 ms        0.80 ms
# host inputs          0.57 K        1.75 ms    ±38.03%        1.68 ms        3.23 ms

# Comparison:
# device inputs        2.74 K
# host inputs          0.57 K - 4.81x slower +1.39 ms
#
# --- AFTER ---

# Operating System: macOS
# CPU Information: Apple M2
# Number of Available Cores: 8
# Available memory: 16 GB
# Elixir 1.20.4
# Erlang 29.1.2

# Benchmark suite executing with the following configuration:
# warmup: 1 s
# time: 3 s
# memory time: 0 ns
# reduction time: 0 ns
# parallel: 8
# inputs: none specified
# Estimated total run time: 8 s

# Benchmarking device inputs ...
# Benchmarking host inputs ...


# Name                    ips        average  deviation         median         99th %
# device inputs       13.50 K       74.07 μs    ±26.93%       68.46 μs      148.08 μs
# host inputs          7.75 K      129.07 μs    ±16.09%      122.71 μs      228.29 μs
#
# Comparison:
# device inputs       13.50 K
# host inputs          7.75 K - 1.74x slower +55.00 μs
# ---
# Name                    ips        average  deviation         median         99th %
# device inputs       11.68 K       85.61 μs    ±33.89%       68.88 μs      157.29 μs
# host inputs          7.69 K      129.99 μs    ±13.92%      125.29 μs      201.08 μs
#
# Comparison:
# device inputs       11.68 K
# host inputs          7.69 K - 1.52x slower +44.37 μs
# ---
# Name                    ips        average  deviation         median         99th %
# device inputs       11.75 K       85.09 μs    ±34.33%       68.67 μs      169.25 μs
# host inputs          7.77 K      128.70 μs    ±15.39%      123.54 μs      212.00 μs
#
# Comparison:
# device inputs       11.75 K
# host inputs          7.77 K - 1.51x slower +43.61 μs
# ---
# Name                    ips        average  deviation         median         99th %
# device inputs       12.37 K       80.86 μs    ±25.43%       74.63 μs      142.70 μs
# host inputs          6.13 K      163.24 μs    ±14.47%      154.04 μs      242.18 μs

# Comparison:
# device inputs       12.37 K
# host inputs          6.13 K - 2.02x slower +82.38 μs

# Run with: mix run bench/argument_decoding.exs
# Small tensors make argument-decoding overhead easier to measure.
inputs = [
  Nx.iota({14}, type: :f64),
  Nx.iota({14}, type: :f64),
  Nx.broadcast(Nx.tensor(1, type: :u8), {14, 14}),
  Nx.tensor(0.0),
  Nx.tensor(0.0)
]

fun =
  EXLA.compile(
    fn left, right, mask, latitude, longitude ->
      Nx.add(
        Nx.add(Nx.sum(left), Nx.sum(right)),
        Nx.add(Nx.sum(mask), Nx.add(latitude, longitude))
      )
    end,
    inputs,
    client: :host
  )

device_inputs = Enum.map(inputs, &Nx.backend_copy(&1, {EXLA.Backend, client: :host}))

Benchee.run(
  %{
    "host inputs" => fn -> apply(fun, inputs) end,
    "device inputs" => fn -> apply(fun, device_inputs) end
  },
  time: 3,
  warmup: 1,
  parallel: 8,
  after_each: &Nx.backend_deallocate/1
)
