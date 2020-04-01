#![allow(dead_code)]

use rand::prelude::*;
use rand::rngs::StdRng;

pub fn main() {
    let mut sort = TinySort::new(4_000, 1_000_000, 100_000_000);

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

#[test]
pub fn test_arithmetic_coding() {
    let mut buf = CircularBitBuffer::new(1_000_000);

    let mut deque = std::collections::VecDeque::new();
    let mut rng = StdRng::seed_from_u64(5);

    let mut encoder = ArithmeticCoder::new(0x288df0c, 101);

    let mut i = 0;
    for _ in 0 .. 100 {
        let r = rng.gen_range(0, 200);
        encoder.encode_number(&mut buf, r);
        deque.push_front(r);
        i += 1;
    }

    let mut decoder = ArithmeticDecoder::new(0x288df0c, 101, &mut buf);

    loop {
        let r = rng.gen_range(0, 200);
        encoder.encode_number(&mut buf, r);
        deque.push_front(r);

        let val = decoder.decode_number(&mut buf);
        let check = deque.pop_back().unwrap();
        if check != val {
            panic!("val = {}, check = {}, i = {}", val, check, i);
        }
        i += 1;
    }
}

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

pub struct TinySort {
    amount: u32,
    maxval: u32,
    boundary: u32,
    minrange: u32,
    buf: Vec<u32>,
    buf_cap: usize,
    storage: CircularBitBuffer,
    committed: usize,
}

impl TinySort {
    pub fn new(sortbuf: usize, amount: u32, maxval: u32) -> TinySort {

        // estimate the amount of space we'll need for this sort. We calculate the theoretically needed and add 0.01% to correct for
        // numerical inaccuracies in the arithmetic encoding
        let ratio = maxval as f64 / amount as f64;
        let required = (amount as f64) * (ratio + 1.0f64).log2() + (maxval as f64) * (1.0 / ratio + 1.0).log2();

        // amount of u32 words needed to contain this, rounded up.
        let bufsize = ((1.0001 * required) as usize) / 32 + 1;
        let mut storage = CircularBitBuffer::new(bufsize);

        let minrange = ratio.ceil() as u32 + 1;
        let boundary = 0xFFFF_FFFFu32 / minrange as u32;

        dbg!(minrange, boundary, bufsize);

        let mut encoder = ArithmeticCoder::new(boundary, minrange);
        encoder.flush(&mut storage);

        TinySort {
            amount,
            maxval,
            boundary,
            minrange,
            buf: Vec::with_capacity(sortbuf),
            buf_cap: sortbuf,
            storage,
            committed: 0,
        }
    }

    pub fn used_space(&self) -> usize {
        self.storage.used_space() + 
        std::mem::size_of::<TinySort>() + 4 * self.buf_cap
    }

    pub fn insert(&mut self, value: u32) {
        self.buf.push(value);

        if self.buf.len() >= self.buf_cap {
            self.commit();
        }
    }

    pub fn commit(&mut self) {
        // sort the temp buffer
        self.buf.sort();

        // deal with spacing in the bitstream
        let storage = &mut self.storage;

        // create our buffer encoder/decoders
        let mut encoder = ArithmeticCoder::new(self.boundary, self.minrange);
        let mut decoder = ArithmeticDecoder::new(self.boundary, self.minrange, storage);

        // merge the temp buffer and the committed buffer into a new committed buffer

        // create the iterators and update the amount of data comitted after this step
        let mut storage_iter = 0 .. self.committed;
        self.committed += self.buf.len();
        let mut new_iter = self.buf.drain(..);

        let mut storage_num = storage_iter.next().map(|_| decoder.decode_number(storage));
        let mut new_num = new_iter.next();

        let mut last_num = 0u32;

        loop {
            match (storage_num, new_num) {
                (None, None) => break,
                (None, Some(n)) => {
                    encoder.encode_number(storage, n - last_num);
                    last_num = n;
                    new_num = new_iter.next();
                },
                (Some(s), Some(n)) if s > n => {
                    encoder.encode_number(storage, n - last_num);
                    last_num = n;
                    new_num = new_iter.next();
                },
                (Some(s), _) => {
                    encoder.encode_number(storage, s - last_num);
                    last_num = s;
                    storage_num = storage_iter.next().map(|_| decoder.decode_number(storage) + s);
                }
            }
        }

        encoder.flush(storage);

        let ratio = self.maxval as f64 / self.amount as f64;
        let theoretical = (self.committed as f64) * (ratio + 1.0f64).log2() + (self.maxval as f64) * (1.0 / ratio + 1.0).log2(); 
        println!("committed {} {} ({})", self.committed, storage.used_space(), theoretical / 8.);
    }
    
    pub fn into_iter(mut self) -> TinySortIterator {
        if self.buf.len() != 0 {
            self.commit();
        }

        TinySortIterator {
            accum: 0,
            decoder: ArithmeticDecoder::new(self.boundary, self.minrange, &mut self.storage),
            inner: self
        }
    }
}

pub struct TinySortIterator {
    inner: TinySort,
    accum: u32,
    decoder: ArithmeticDecoder
}

impl Iterator for TinySortIterator {
    type Item = u32;
    fn next(&mut self) -> Option<u32> {
        if self.inner.committed == 0 {
            return None;
        }
        self.inner.committed -= 1;

        let decoded = self.decoder.decode_number(&mut self.inner.storage);
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
    fn encode(&mut self, bitstream: &mut CircularBitBuffer, bit: bool) {
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
    fn encode_number(&mut self, bitstream: &mut CircularBitBuffer, value: u32) {
        for _ in 0 .. value {
            self.encode(bitstream, true);
        }
        self.encode(bitstream, false);
    }
    fn flush(&mut self, bitstream: &mut CircularBitBuffer) {
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
    pub fn new(boundary: u32, minrange: u32, bitstream: &mut CircularBitBuffer) -> ArithmeticDecoder {
        ArithmeticDecoder {
            bottom: 0,
            range: 0xFFFF_FFFF,
            boundary,
            minrange,
            workbuf: bitstream.pull_bits(32).unwrap(),
        }
    }
    fn decode(&mut self, bitstream: &mut CircularBitBuffer) -> bool {
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
    fn decode_number(&mut self, bitstream: &mut CircularBitBuffer) -> u32 {
        let mut i = 0;
        while self.decode(bitstream) {
            i += 1;
        }
        i
    }
}
