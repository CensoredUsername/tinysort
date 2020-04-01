#![allow(dead_code)]

use rand::prelude::*;
//use rand::rngs::StdRng;

pub fn main() {
    let mut sort = TinySort::new(8_000, 1_000_000, 100_000_000);

    let mut values = generate_random_values(1_000_000, 100_000_000);

    for value in values.iter().cloned() {
        sort.insert(value);
    }

    println!("used {}", sort.used_space());

    let sorted: Vec<u32> = sort.into_iter().collect();

    values.sort();
    assert!(values == sorted);
}

/// Generates a buffer of 1 million values of maximally 8 decimal digits
pub fn generate_random_values(amount: usize, max: u32) -> Vec<u32> {
    let mut buf = Vec::new();
    let mut rng = thread_rng();
    for _ in 0 .. amount {
        buf.push(rng.gen_range(0, max));
    }
    buf
}

// #[test]
// pub fn test_arithmetic_coding() {
//     let mut buf = CircularBitBuffer::new(1_000_000);

//     let mut deque = std::collections::VecDeque::new();
//     let mut rng = StdRng::seed_from_u64(5);

//     let mut encoder = ArithmeticCoder::new(0x288df0c, 101);

//     let mut i = 0;
//     for _ in 0 .. 100 {
//         let r = rng.gen_range(0, 200);
//         encoder.encode_number(&mut buf, r);
//         deque.push_front(r);
//         i += 1;
//     }

//     let mut decoder = ArithmeticDecoder::new(0x288df0c, 101, &mut buf);

//     loop {
//         let r = rng.gen_range(0, 200);
//         encoder.encode_number(&mut buf, r);
//         deque.push_front(r);

//         let val = decoder.decode_number(&mut buf);
//         let check = deque.pop_back().unwrap();
//         if check != val {
//             panic!("val = {}, check = {}, i = {}", val, check, i);
//         }
//         i += 1;
//     }
// }

pub struct CircularBitBuffer {
    buf: Vec<u32>,
    head: usize,
    tail: usize
}

impl CircularBitBuffer {
    pub fn new(buf: usize) -> CircularBitBuffer {
        CircularBitBuffer {
            buf: vec![0; buf],
            head: 0,
            tail: 0
        }
    }

    pub fn used_space(&self) -> usize {
        std::mem::size_of::<CircularBitBuffer>() + (self.len() + 3) / 8
    }

    pub fn len(&self) -> usize {
        let mut head = self.head;
        if head < self.tail {
            head += self.buf.len() * 32;
        }
        head - self.tail
    }

    pub fn pull(&mut self) -> Option<bool> {
        // check if we have some values
        if self.tail == self.head {
            return None;
        }
        // get the value
        let rv = self.buf[self.tail / 32] & (1 << (self.tail % 32)) != 0;
        // increment tail
        self.tail += 1;
        if self.tail == self.buf.len() * 32 {
            self.tail = 0;
        }
        Some(rv)
    }

    pub fn push(&mut self, bit: bool) -> Result<(), ()> {
        // generate an incremented head
        let mut head = self.head + 1;
        if head == self.buf.len() * 32 {
            head = 0;
        }
        // see if that would cause us to run into tail
        if head == self.tail {
            return Err(())
        }

        // edit the bit
        let mut val = self.buf[self.head / 32];
        val &= !(1 << (self.head % 32));
        val |= (bit as u32) << (self.head % 32);
        self.buf[self.head / 32] = val;

        // increment head
        self.head = head;
        Ok(())
    }

    pub fn pull_bits(&mut self, bits: u8) -> Option<u32> {
        let mut accum = 0;

        for _ in 0 .. bits {
            accum <<= 1;
            accum |= self.pull()? as u32;
        }

        Some(accum)
    }

    pub fn push_bits(&mut self, bits: u8, value: u32) -> Result<(), ()> {
        let mut i = bits;
        while i != 0 {
            i -= 1;
            self.push(value & (1 << i) != 0)?;
        }
        Ok(())
    }
}

pub struct BitStreamReadWriter<'a> {
    buf: &'a mut [u32],

    // the idx we'll next read from
    readidx: usize,
    // the idx we'll next write to
    writeidx: usize,
}

impl<'a> BitStreamReadWriter<'a> {
    pub fn new(buf: &mut [u32], readidx: usize, writeidx: usize) -> BitStreamReadWriter {
        BitStreamReadWriter {
            buf,
            readidx,
            writeidx,
        }
    }

    pub fn get_writeidx(&self) -> usize {
        self.writeidx
    }

    pub fn get_readidx(&self) -> usize {
        self.readidx
    }

    pub fn pull(&mut self) -> Option<bool> {
        // check if we wouldn't cross the other idx and get corrupt data
        if self.readidx + 1 == self.writeidx {
            return None;
        }

        // read the bit
        let rv = self.buf[self.readidx / 32] & (1 << (self.readidx % 32)) != 0;

        self.readidx += 1;

        Some(rv)
    }

    pub fn push(&mut self, bit: bool) -> Result<(), ()> {
        // check if we wouldn't cross the other idx and create corrupt data
        if self.writeidx + 1 == self.readidx {
            return Err(())
        }

        // edit the bit we're at
        let mut val = self.buf[self.writeidx / 32];
        val &= !(1 << (self.writeidx % 32));
        val |= (bit as u32) << (self.writeidx % 32);
        self.buf[self.writeidx / 32] = val;

        self.writeidx += 1;

        Ok(())
    }

    pub fn pull_bits(&mut self, bits: u8) -> Option<u32> {
        let mut accum = 0;

        for _ in 0 .. bits {
            accum <<= 1;
            accum |= self.pull()? as u32;
        }

        Some(accum)
    }

    pub fn push_bits(&mut self, bits: u8, value: u32) -> Result<(), ()> {
        let mut i = bits;
        while i != 0 {
            i -= 1;
            self.push(value & (1 << i) != 0)?;
        }
        Ok(())
    }
}

pub struct TinySort {
    // core algorithm paramters: the amount of values to store, the maximum value to store and the amount of extra
    // space to use
    amount: u32,
    maxval: u32,
    extra: usize,

    // Arithmetic coding precalculated constants: the boundary between a 0 and a 1, and the mimimum
    // range required to not lose numerical precision
    boundary: u32,
    minrange: u32,

    // the buffer we use to store all our data in. compressed data or intermediate stuff to be sorted
    buf: Vec<u32>,

    // the amount of values committed to compression
    committed: usize,
    // the size of the committed buffer, in words
    committed_len: usize,

    // amount of u32's reserved for intermediate sorting
    sort_cap: usize,
    // amount of u32 collected but not sorted yet
    sort_pending: usize,
}

impl TinySort {
    fn calc_new_sort_cap(&mut self) {
        let mut sort_cap = self.extra;
        let ratio = self.maxval as f64 / self.amount as f64;
        loop {
            let new_sort_cap = sort_cap + self.extra;
            let next_committed = self.committed + new_sort_cap;
            let theoretical_bits_required = (next_committed as f64) * (ratio + 1.0f64).log2() + (self.maxval as f64) * (1.0 / ratio + 1.0).log2();
            let words_required = ((1.0001 * theoretical_bits_required) as usize) / 32 + 5;

            if words_required + new_sort_cap > self.buf.len() {
                //println!("Going to collect {} values. This leaves {} space for the compressed buffers", sort_cap, 4 * (self.buf.len() - sort_cap));
                break;
            }
            sort_cap = new_sort_cap;
        }
        // best possible sorting capacity
        self.sort_cap = sort_cap;
    }

    pub fn new(extra: usize, amount: u32, maxval: u32) -> TinySort {

        // estimate the amount of space we'll need for this sort. We calculate the theoretically needed and add 0.01% to correct for
        // numerical inaccuracies in the arithmetic encoding
        let ratio = maxval as f64 / amount as f64;
        let required = (amount as f64) * (ratio + 1.0f64).log2() + (maxval as f64) * (1.0 / ratio + 1.0).log2();

        // amount of u32 words needed to contain this, rounded up.
        let bufsize = ((1.0001 * required) as usize) / 32 + 5 + extra;

        let buf = vec![0; bufsize];

        let minrange;
        let boundary;
        if ratio >= 1. {
            minrange = ratio.ceil() as u32 + 1;
            boundary = 0xFFFF_FFFFu32 / minrange as u32;
        } else {
            minrange = (1. / ratio).ceil() as u32 + 1;
            boundary = 0xFFFF_FFFF - 0xFFFF_FFFFu32 / minrange as u32;
        }

        dbg!(minrange, boundary, bufsize*4);

        // build a fake "committed" section for now
        let committed = 0;
        let committed_len = 1;

        let mut rv = TinySort {
            amount,
            maxval,
            extra,

            boundary,
            minrange,

            buf,

            committed,
            committed_len,

            sort_cap: extra,
            sort_pending: 0,
        };
        rv.calc_new_sort_cap();
        rv
    }

    pub fn used_space(&self) -> usize {
        std::mem::size_of::<TinySort>() + 4 * self.buf.len()
    }

    pub fn insert(&mut self, value: u32) {
        let idx = self.buf.len() - self.sort_cap + self.sort_pending;
        self.buf[idx] = value;
        self.sort_pending += 1;

        if self.sort_pending == self.sort_cap {
            self.commit();
        }
    }

    pub fn commit(&mut self) {
        // split the buffer
        let idx = self.buf.len() - self.sort_cap;
        let (compressed_buf, mut presort_buf) = self.buf.split_at_mut(idx);

        // sort all pending values
        presort_buf = &mut presort_buf[..self.sort_pending];
        presort_buf.sort_unstable();

        // right-align the old compressed buffer
        compressed_buf.copy_within(0 .. self.committed_len, idx - self.committed_len);

        // create the bitstream reader
        let mut storage = BitStreamReadWriter::new(compressed_buf, (idx - self.committed_len) * 32, 0);

        // create our buffer encoder/decoders
        let mut encoder = ArithmeticCoder::new(self.boundary, self.minrange);
        let mut decoder = ArithmeticDecoder::new(self.boundary, self.minrange, &mut storage);

        //println!("Starting commit. compressed_len={}, compressed={}. compressed buffer relocated to {}", self.committed_len, self.committed, (idx - self.committed_len));

        // merge the temp buffer and the committed buffer into a new committed buffer

        // create the iterators and update the amount of data comitted after this step
        let mut storage_iter = 0 .. self.committed;
        let mut new_iter = presort_buf.iter().cloned();

        let mut storage_num = storage_iter.next().map(|_| decoder.decode_number(&mut storage));
        let mut new_num = new_iter.next();

        let mut last_num = 0u32;

        loop {
            match (storage_num, new_num) {
                (None, None) => break,
                (None, Some(n)) => {
                    encoder.encode_number(&mut storage, n - last_num);
                    last_num = n;
                    new_num = new_iter.next();
                },
                (Some(s), Some(n)) if s > n => {
                    encoder.encode_number(&mut storage, n - last_num);
                    last_num = n;
                    new_num = new_iter.next();
                },
                (Some(s), _) => {
                    encoder.encode_number(&mut storage, s - last_num);
                    last_num = s;
                    storage_num = storage_iter.next().map(|_| decoder.decode_number(&mut storage) + s);
                }
            }
        }

        encoder.flush(&mut storage);

        // update values
        self.committed += self.sort_pending;
        self.sort_pending = 0;
        self.committed_len = (storage.get_writeidx() + 31) / 32;

        let ratio = self.maxval as f64 / self.amount as f64;
        let theoretical = (self.committed as f64) * (ratio + 1.0f64).log2() + (self.maxval as f64) * (1.0 / ratio + 1.0).log2(); 
        println!("committed {} {} ({})", self.committed, self.committed_len * 4, theoretical / 8.);

        self.calc_new_sort_cap();
    }
    
    pub fn into_iter(mut self) -> TinySortIterator {
        if self.sort_pending != 0 {
            self.commit();
        }

        let mut storage = BitStreamReadWriter::new(&mut self.buf, 0, 0);
        let decoder = ArithmeticDecoder::new(self.boundary, self.minrange, &mut storage);
        let readidx = storage.get_readidx();

        TinySortIterator {
            buf: self.buf,
            committed: self.committed,
            accum: 0,
            decoder,
            readidx,
        }
    }
}

pub struct TinySortIterator {
    buf: Vec<u32>,
    committed: usize,
    accum: u32,
    decoder: ArithmeticDecoder,
    readidx: usize,
}

impl Iterator for TinySortIterator {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        if self.committed == 0 {
            return None;
        }

        self.committed -= 1;

        let mut storage = BitStreamReadWriter::new(&mut self.buf, self.readidx, 0);
        let decoded = self.decoder.decode_number(&mut storage);
        self.readidx = storage.get_readidx();
        self.accum += decoded;
        Some(self.accum)
    }
}


#[derive(Debug, Clone)]
struct ArithmeticCoder {
    bottom: u32,
    range: u64,
    boundary: u32,
    minrange: u32,
}

impl ArithmeticCoder {
    pub fn new(boundary: u32, minrange: u32) -> ArithmeticCoder {
        ArithmeticCoder {
            bottom: 0,
            range: 0xFFFF_FFFF,
            boundary,
            minrange
        }
    }
    fn encode(&mut self, bitstream: &mut BitStreamReadWriter, bit: bool) {
        let mut a: u32;
        let mut b: u32;

        if !bit {
            a = self.bottom;
            b = self.bottom + ((self.range * (self.boundary as u64)) >> 32) as u32;
        } else {
            a = self.bottom + 1 + ((self.range * (self.boundary as u64)) >> 32) as u32;
            b = self.bottom + self.range as u32;
        }

        while (a ^ b) & 0x8000_0000 == 0 {
            bitstream.push(a & 0x8000_0000 != 0).unwrap();
            a = a << 1;
            b = (b << 1) | 1;
        }
        if (b - a) <= self.minrange {
            // we get here if the algorithm is unable to shift extra bits out. in this case 
            // A will be something like 0b011111111111111 and 
            // B will be something like 0b100000000000000.
            // that's annoying and suboptimal. bump one of them over the step so we avoid this signularity
            // while staying in range
            if (b - 0x7FFF_FFFF) < (0x8000_0000 - a) {
                b = 0x7FFF_FFFF;
            } else {
                a = 0x8000_0000;
            }

            while (a ^ b) & 0x8000_0000 == 0 {
                bitstream.push(a & 0x8000_0000 != 0).unwrap();
                a = a << 1;
                b = (b << 1) | 1;
            }
        }


        self.bottom = a;
        self.range = (b - a) as u64;
    }
    fn encode_number(&mut self, bitstream: &mut BitStreamReadWriter, value: u32) {
        for _ in 0 .. value {
            self.encode(bitstream, true);
        }
        self.encode(bitstream, false);
    }
    fn flush(&mut self, bitstream: &mut BitStreamReadWriter) {
        bitstream.push_bits(32, self.bottom + (self.range as u32)/ 2).unwrap();
    }
}

#[derive(Debug, Clone)]
struct ArithmeticDecoder {
    bottom: u32,
    range: u64,
    boundary: u32,
    minrange: u32,
    workbuf: u32
}

impl ArithmeticDecoder {
    pub fn new(boundary: u32, minrange: u32, bitstream: &mut BitStreamReadWriter) -> ArithmeticDecoder {
        ArithmeticDecoder {
            bottom: 0,
            range: 0xFFFF_FFFF,
            boundary,
            minrange,
            workbuf: bitstream.pull_bits(32).unwrap(),
        }
    }
    fn decode(&mut self, bitstream: &mut BitStreamReadWriter) -> bool {
        let mut a: u32;
        let mut b: u32;

        a = self.bottom;
        b = self.bottom + ((self.range * (self.boundary as u64)) >> 32) as u32;

        let rv = self.workbuf > b;

        if rv {
            a = b + 1;
            b = self.bottom + self.range as u32;
        }

        while (a ^ b) & 0x8000_0000 == 0 {
            a = a << 1;
            b = (b << 1) | 1;
            self.workbuf = (self.workbuf << 1) | (bitstream.pull().unwrap() as u32);
        }
        if (b - a) <= self.minrange {
            // we get here if the algorithm is unable to shift extra bits out. in this case 
            // A will be something like 0b011111111111111 and 
            // B will be something like 0b100000000000000.
            // that's annoying and suboptimal. bump one of them over the step so we avoid this signularity
            // while staying in range
            if (b - 0x7FFF_FFFF) < (0x8000_0000 - a) {
                b = 0x7FFF_FFFF;
            } else {
                a = 0x8000_0000;
            }

            while (a ^ b) & 0x8000_0000 == 0 {
                a = a << 1;
                b = (b << 1) | 1;
                self.workbuf = (self.workbuf << 1) | (bitstream.pull().unwrap() as u32);
            }
        }

        self.bottom = a;
        self.range = (b - a) as u64;

        rv
    }
    fn decode_number(&mut self, bitstream: &mut BitStreamReadWriter) -> u32 {
        let mut i = 0;
        while self.decode(bitstream) {
            i += 1;
        }
        i
    }
}
