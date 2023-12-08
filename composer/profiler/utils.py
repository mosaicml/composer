from base64 import b64encode
from os import remove
from tempfile import NamedTemporaryFile
import numpy as np
from torch.profiler._memory_profiler import MemoryProfileTimeline, _CATEGORY_TO_COLORS, _CATEGORY_TO_INDEX
import importlib.util
from torch.profiler.profiler import profile as TorchProfile
from typing import Optional
import torch.cuda


def export_memory_timeline_html(prof: TorchProfile,
                                path: str,
                                device: Optional[str | int] = None,
                                figsize=(20, 12),
                                title=None) -> None:
    # Default to device 0, if unset. Fallback on cpu.
    if device is None and prof.use_device and prof.use_device != "cuda":
        device = prof.use_device + ":0"

    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Construct the memory timeline plot data
    mem_tl = MemoryProfileTimeline(prof._memory_profile())

    # Check if user has matplotlib installed, return gracefully if not.
    matplotlib_spec = importlib.util.find_spec('matplotlib')
    if matplotlib_spec is None:
        print('export_memory_timeline_html failed because matplotlib was not found.')
        return
    import matplotlib.pyplot as plt

    mt = mem_tl._coalesce_timeline(device)
    times, sizes = np.array(mt[0]), np.array(mt[1])
    stacked = np.cumsum(sizes, axis=1) / 1024**3
    max_memory_allocated = torch.cuda.max_memory_allocated()
    max_memory_reserved = torch.cuda.max_memory_reserved()

    # Plot memory timeline as stacked data
    fig = plt.figure(figsize=figsize, dpi=80)
    axes = fig.gca()
    for category, color in _CATEGORY_TO_COLORS.items():
        i = _CATEGORY_TO_INDEX[category]
        axes.fill_between(times / 1e3, stacked[:, i], stacked[:, i + 1], color=color, alpha=0.7)
    fig.legend(['Unknown' if i is None else i.name for i in _CATEGORY_TO_COLORS])
    axes.set_xlabel('Time (us)')
    axes.set_ylabel('Memory (GB)')
    _, end = axes.get_ylim()
    axes.grid(True)
    axes.set_yticks(np.arange(0, end, 1))
    title = '\n\n'.join(([title] if title else []) + [
        f'Max memory allocated: {max_memory_allocated/(10**9):.2f} GB \n'
        f'Max memory reserved: {max_memory_reserved/(10**9):.2f} GB'
    ])
    axes.set_title(title)

    # Embed the memory timeline image into the HTML file
    tmpfile = NamedTemporaryFile('wb', suffix='.png', delete=False)
    tmpfile.close()
    fig.savefig(tmpfile.name, format='png')

    with open(tmpfile.name, 'rb') as tmp:
        encoded = b64encode(tmp.read()).decode('utf-8')
        html = f"""<html>
                <head><meta charset="utf-8" /><title>GPU Memory Timeline HTML</title></head>
                <body>
                <img src='data:image/png;base64,{encoded}'>
                </body>
                </html>"""

        with open(path, 'w') as f:
            f.write(html)
    remove(tmpfile.name)
