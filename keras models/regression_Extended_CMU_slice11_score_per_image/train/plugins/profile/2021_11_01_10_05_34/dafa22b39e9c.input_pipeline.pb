  *??C??@)      ?=2u
>Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Mapa6??G*@!???!?W@)
If?*@1YM?w0?W@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapx'???!?|Fl&d@)i? ?w???1????@:Preprocessing2?
PIterator::Model::ParallelMapV2::Zip[0]::FlatMap[11]::Concatenate[0]::TensorSlice'/2?F??!wd???W??)'/2?F??1wd???W??:Preprocessing2?
LIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeatX?vMHk??!?48Ȩx??)?!?aK???1???=???:Preprocessing2F
Iterator::Model%#gaO;??!??I??]??)?Gp#e???1aӧq????:Preprocessing2w
@Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[10]::Concatenate?֥F?g??!??q???)"U?????1@W9?#??:Preprocessing2p
9Iterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetcht??????!??R?u??)t??????1??R?u??:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?J!?K??!)????)^f?(???1t??x???:Preprocessing2U
Iterator::Model::ParallelMapV2r??>s֗?!7?C?X???)r??>s֗?17?C?X???:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??g????!??/S?@)?Mc{-???1?Z(????:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensorHo??ܚ??!h??.????)Ho??ܚ??1h??.????:Preprocessing2?
SIterator::Model::ParallelMapV2::Zip[0]::FlatMap::Prefetch::Map::FiniteRepeat::Rangeꗈ?ο}?!x,??X???)ꗈ?ο}?1x,??X???:Preprocessing2w
@Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[11]::ConcatenateӾ????!Gj?*???)??7h?>n?1t\??:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[10]::Concatenate[1]::FromTensor?30??&V?!{?Wl?	??)?30??&V?1{?Wl?	??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisg
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*no#You may skip the rest of this page.BX
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.