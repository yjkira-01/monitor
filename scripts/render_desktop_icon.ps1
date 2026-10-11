$ErrorActionPreference='Stop'
Add-Type -AssemblyName PresentationCore,PresentationFramework,WindowsBase
$appRoot=Join-Path (Split-Path $PSScriptRoot -Parent) 'desktop/CareMonitor'
$reader=[Xml.XmlReader]::Create((Join-Path $appRoot 'Themes/Icons.xaml'))
try { $resources=[Windows.Markup.XamlReader]::Load($reader) } finally { $reader.Dispose() }
$brand=$resources['AppMark']
$brand.Freeze()
function Render-Brand([int]$size) {
    $visual=[Windows.Media.DrawingVisual]::new()
    $drawing=$visual.RenderOpen()
    # Leave transparent breathing room around the tile at every Windows icon size.
    $inset=[Math]::Max(1,[Math]::Round($size*0.055))
    $drawing.DrawImage($brand,[Windows.Rect]::new($inset,$inset,$size-2*$inset,$size-2*$inset))
    $drawing.Close()
    $bitmap=[Windows.Media.Imaging.RenderTargetBitmap]::new($size,$size,96,96,[Windows.Media.PixelFormats]::Pbgra32)
    $bitmap.Render($visual)
    $encoder=[Windows.Media.Imaging.PngBitmapEncoder]::new()
    $encoder.Frames.Add([Windows.Media.Imaging.BitmapFrame]::Create($bitmap))
    $stream=[IO.MemoryStream]::new()
    $encoder.Save($stream)
    $data=$stream.ToArray()
    $stream.Dispose()
    return ,$data
}
$sizes=@(16,20,24,32,40,48,64,128,256)
$pngs=@($sizes | ForEach-Object { ,(Render-Brand $_) })
$stream=[IO.MemoryStream]::new()
$writer=[IO.BinaryWriter]::new($stream)
$writer.Write([uint16]0);$writer.Write([uint16]1);$writer.Write([uint16]$sizes.Count)
$offset=6+16*$sizes.Count
for($i=0;$i -lt $sizes.Count;$i++) {
    $dimension=if($sizes[$i] -eq 256){0}else{$sizes[$i]}
    $writer.Write([byte]$dimension);$writer.Write([byte]$dimension)
    $writer.Write([byte]0);$writer.Write([byte]0)
    $writer.Write([uint16]1);$writer.Write([uint16]32)
    $writer.Write([uint32]$pngs[$i].Length);$writer.Write([uint32]$offset)
    $offset+=$pngs[$i].Length
}
foreach($png in $pngs){$writer.Write([byte[]]$png)}
$writer.Flush()
[IO.File]::WriteAllBytes((Join-Path $appRoot 'Assets/AppIcon.ico'),$stream.ToArray())
[IO.File]::WriteAllBytes((Join-Path $appRoot 'Assets/AppIcon.png'),$pngs[-1])
$writer.Dispose();$stream.Dispose()
[IO.File]::WriteAllBytes((Join-Path (Split-Path $PSScriptRoot -Parent) 'artifacts/desktop_app_icon.png'),(Render-Brand 512))
Write-Output 'Generated transparent PNG and nine-resolution Windows ICO from the native vector mark.'
