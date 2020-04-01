# Tinysort

Tinysort is a library for sorting data in as little memory as possible.

## Example

```rust
use tinysort::TinySort;

let mut sort = TinySort::new(8000, 1_000_000, 100_000_000).unwrap();

// create some values to sort between the indicated borders
let mut values: Vec<u32> = (0 .. 1_000_000).map(|i| (100_000_000.0 * (i as f64 * 0.4567895678).fract()) as u32).collect();
for value in values.iter().cloned() {
     sort.insert(value);
}

println!("TinySort using {} bytes of memory, normal sort using {} bytes of memory.", sort.used_space(), values.len() * std::mem::size_of::<u32>());
let sorted: Vec<u32> = sort.into_iter().collect();
values.sort_unstable();
assert!(sorted == values);
```
This example will print `TinySort using 1043916 bytes of memory, normal sort using 4000000 bytes of memory.`,
showing the almost 4x memory usage drop thanks to TinySort.

## Explanation

With sorting algorithms, there is often a trade-off between speed and memory usage.
For classical sorting algorithms, we count memory usage as the extra memory usage of the algorithm, meaning any space the algorithm
needs for storing the data it is operating on on top of the storage of the original values.

Tinysort, under the applicable cases, does not need any extra memory. In fact, it requires **less** memory to store all sorted values than it would just take to store them in a contiguous array. It only requires as much memory as is necessary to store the raw entropy of the contained sorted values with a small margin for bookkeeping purposes.

This means it can sort and store a million numbers between 0 and 100 million in less than a MiB of memory.

## Complexity

The memory and time complexity of this algorithm are dependent on two properties of its input. The first being the total amount of values it will process (`N`) and the second being the largest value contained in the input (`M`)

The memory use of the algorithm scales as worst case `O(N*log(1+M/N) + M*log(1+N/M))` (which deserves some reward for ridiculousness)

The time complexity of the algorithm is worst case `O((N + M)*(Log(N)))`

## Internal details

The algorithm stores the sorted data as a arithmetic coded bitstream in a circular bit buffer. It accumulates an amount of given values, sorts them with a normal sorting algorithm and then merges these values with a streaming decode of the currently stored sorted values, immediately writing the new sorted stream after it in the circular buffer. The amount of values to accumulate is chosen based on the space left over in the buffer after these values would be compressed.

To store all numbers this compact, the sorting algorithmm does not store the actual numbers in the compressed stream. It stores a list of differences in this compressed stream. These are then compressed via arithmetic coding.

The chosen arithmetic code model has an alphabet of two letters. 1 means to add 1 to the accumulator, 0 means emit the current number. This means that in effect, all numbers are first transformed to a unary representation of n 1's, with a terminating 0. As the amount of 1's will equal the maximum number that was sorted, and the amount of 0's will match the amount of numbers totally sorted, this allows the algorithm to calculate the most optimal probability distribution of the alphabet, and use an arithmetic code model that comes extremely close to the theoretically maximum value, even accounting for inefficiencies of the algorithm as it operates in fixed precision.

## FAQ

**Q: How is this even possible.**

A: The amount of entropy required to pick an amount of values out of a set, with no regard for order, is significantly lower than the amount of entropy required to store these values ordered.

**Q: This algorithm is slow.**

A: It is as fast as it can be, under these memory constraints.

**Q: Why would you ever want to use this.**

A: We do what we must, because we can.

## Usage

Todo: currently the project is not a full library yet, just change the values in `main` to see how it performs. Be prepared to get some coffee for larger values though.

## Contributing

If you're insane enough to think that this is a good idea to ever use, your contributions are of course welcome.

## License

[MPL2.0](https://www.mozilla.org/en-US/MPL/2.0/)
