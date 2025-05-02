for file in *.py; do
  sed -i '' '/data:image\//d' "$file"
done