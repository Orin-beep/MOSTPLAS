# coding = utf-8

def convert_kmer_reverse_complement(sequence, k):
    nucleobase = ['A', 'G', 'C', 'T']
    kmer_frequency = {}

    def kmer_generation(nums, k):
        def backtracking(nums, k):
            if len(path) == k:
                result.append(path[:])
                return
            for i in range(len(nums)):
                path.append(nums[i])
                backtracking(nums, k)
                path.pop()

        result = []
        path = []
        backtracking(nums, k)
        return result

    kmers = kmer_generation(nucleobase, k)
    for kmer in kmers:
        string = ''
        for i in range(len(kmer)):
            string += kmer[i]
        string_reverse = Seq(string).reverse_complement()
        if string not in kmer_frequency and string_reverse not in kmer_frequency:
            kmer_frequency[string] = 0

    for i in range(len(sequence) - k + 1):
        string = sequence[i:i + k]
        string_reverse = Seq(string).reverse_complement()
        if string in kmer_frequency:
            kmer_frequency[string] += 1
        elif string_reverse in kmer_frequency:
            kmer_frequency[string_reverse] += 1

    sum_frequency = sum(kmer_frequency.values())
    if sum_frequency == 0:
        return kmer_frequency
    for key, value in kmer_frequency.items():
        kmer_frequency[key] = float(value / sum_frequency)
    return kmer_frequency