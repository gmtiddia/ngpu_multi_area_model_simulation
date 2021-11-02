#!/bin/bash
rm -f data[0-9]
rm -rf */spikes_pop_idx
rm -f */pop_list.txt
i=0
for fn in $(ls */recordings/*TH_6I.dat); do
    dn=$(echo $fn | tr '/' ' ' | awk '{print $1}')
    echo $dn 
    currdir=$(pwd)
    ln -s ${currdir}/$dn ${currdir}/data${i}
    j=0
    :>$dn/pop_list.txt
    mkdir -p $dn/spikes_pop_idx
    :>$dn/spikes_pop_idx/population_nodeids.dat
    cat $dn/recordings/network_gids.txt | tr ',' ' ' | while read area pop i0 i1; do
	echo "run $i pop $j: $area $pop" 
	echo "$j $area $pop" >> $dn/pop_list.txt
	echo "$i0 $i1" >> $dn/spikes_pop_idx/population_nodeids.dat
	ln -s ${currdir}/$dn/recordings/spike_times_${area}_${pop}.dat ${currdir}/$dn/spikes_pop_idx/spike_times_$j.dat
	#echo "sender time_ms" > $dn/spikes_pop_idx/spike_times_$j.dat
        #cat $dn/spikes_pop/spikes_${area}_${pop}.dat >> $dn/spikes_pop_idx/spike_times_$j.dat
	j=$(expr $j + 1)
    done
    i=$(expr $i + 1)
done
