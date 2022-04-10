#!/bin/bash
rm -f data[0-9]
i=0
for fn in $(ls */recordings/spike_times_TH_6I.dat); do
    dn=$(echo $fn | tr '/' ' ' | awk '{print $1}')
    echo $dn
    currdir=$(pwd)
    ln -s ${currdir}/$dn ${currdir}/data${i}
    j=0
    mkdir -p $dn/spikes_pop_idx
    :>$dn/pop_list.txt
    :>$dn/spikes_pop_idx/population_nodeids.dat
    cat $dn/recordings/network_gids.txt | tr ',' ' ' | while read area pop i0 i1; do
	echo "run $i pop $j: $area $pop" 
	echo "$j $area $pop" >> $dn/pop_list.txt
	echo "$i0 $i1" >> $dn/spikes_pop_idx/population_nodeids.dat
	tail -n +2 ${currdir}/$dn/recordings/spike_times_${area}_${pop}.dat > ${currdir}/$dn/spikes_pop_idx/spike_times_$j.dat
	j=$(expr $j + 1)
    done
    i=$(expr $i + 1)
done
