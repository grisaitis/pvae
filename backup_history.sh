history | cut -c 8- | awk '!seen[$0]++' > history_export.txt