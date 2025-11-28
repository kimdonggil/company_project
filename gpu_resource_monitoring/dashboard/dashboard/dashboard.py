import reflex as rx
from typing import List, Dict
import httpx

gpu_memory_0 = 543
gpu_percent_0 = 2.21

gpu_memory_1 = 239
gpu_percent_1 = 2.93

class State(rx.State):
    gpu_count: int = 0
    mem_rates: List[Dict] = []

    GPU_API_URLS = [
        "http://10.40.217.244:8123/gpu/",
        "http://10.40.217.187:8123/gpu/"
    ]

    def fetch_gpu_info(self):
        all_mem = []
        try:
            with httpx.Client() as client:
                for url in self.GPU_API_URLS:
                    try:
                        resp = client.get(url, timeout=2.0)
                        resp.raise_for_status()
                        gpus = resp.json()
                        for gpu in gpus:
                            all_mem.append({
                                "gpu_index": gpu["index"],
                                "mem_used": gpu["mem_used"],
                                "mem_total": gpu["mem_total"],
                                "mem_used_percent": gpu["mem_used_percent"],
                            })
                    except Exception as e:
                        print(f"API 요청 실패: {url}, {e}")

            self.set(mem_rates=all_mem, gpu_count=len(all_mem))
        except:
            self.set(mem_rates=[], gpu_count=0)

    # NVIDIA RTX 3090 GPU 0
    @rx.var
    def mem_gpu_indexs_0(self) -> str:
        return self.mem_rates[0]["gpu_index"] if len(self.mem_rates) > 0 else 0
    @rx.var
    def mem_useds_0(self) -> int:
        return self.mem_rates[0]["mem_used"] if len(self.mem_rates) > 0 else 0
    @rx.var
    def mem_totals_0(self) -> int:
        return self.mem_rates[0]["mem_total"] if len(self.mem_rates) > 0 else 0
    @rx.var
    def mem_used_percents_0(self) -> float:
        return self.mem_rates[0]["mem_used_percent"] if len(self.mem_rates) > 0 else 0.0
    @rx.var
    def mem_status_0(self) -> str:
        if len(self.mem_rates) > 0:
            used_percent = self.mem_rates[0]["mem_used_percent"]
            return "사용 중" if used_percent > gpu_percent_0 else "사용 전"
        return "사용 전"
    @rx.var
    def mem_used_percents_0_fmt(self) -> str:
        if len(self.mem_rates) > 0:
            val = round(self.mem_rates[0]['mem_used_percent']-gpu_percent_0, 3)
            return str(val).rstrip('0').rstrip('.')
        return "0"

    # NVIDIA GTX 1080 GPU 0
    @rx.var
    def mem_gpu_indexs_1(self) -> str:
        return self.mem_rates[1]["gpu_index"] if len(self.mem_rates) > 1 else 0
    @rx.var
    def mem_useds_1(self) -> int:
        return self.mem_rates[1]["mem_used"] if len(self.mem_rates) > 1 else 0
    @rx.var
    def mem_totals_1(self) -> int:
        return self.mem_rates[1]["mem_total"] if len(self.mem_rates) > 1 else 0
    @rx.var
    def mem_used_percents_1(self) -> float:
        return self.mem_rates[1]["mem_used_percent"] if len(self.mem_rates) > 1 else 0.0
    @rx.var
    def mem_status_1(self) -> str:
        if len(self.mem_rates) > 0:
            used_percent = self.mem_rates[1]["mem_used_percent"]
            return "사용 중" if used_percent > gpu_percent_1 else "사용 전"
        return "사용 전"
    @rx.var
    def mem_used_percents_1_fmt(self) -> str:
        if len(self.mem_rates) > 1:
            val = round(self.mem_rates[1]['mem_used_percent']-gpu_percent_1, 3)
            return str(val).rstrip('0').rstrip('.')
        return "0"    

def index() -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.vstack(
                rx.hstack(
                    rx.heading("GPU 자원 모니터링", size="3em"),
                    rx.color_mode.button(style={"opacity": "1", "scale": "1"}),
                    align_items="center",
                    margin_bottom="1.2em",
                    margin_top="1.8em",
                ),
                rx.badge(rx.text(rx.text.strong(rx.moment(interval=1000, format='YYYY-MM-DD HH:mm:ss'))), color_scheme="orange", size="3", variant='surface'),
                rx.box(height="1em"),
                spacing="0.5em",
                width="300px",
            ),            
            rx.vstack(
                rx.moment(
                    interval=1000,
                    on_change=lambda date: State.fetch_gpu_info(),
                    style={"display": "none"}
                ),


                rx.flex(
                    rx.card(
                        rx.flex(
                            rx.box(
                                rx.hstack(
                                    rx.text(
                                        "GPU 노드 개수",
                                        rim="bold",
                                        font_size="3rem",
                                        weight="both"
                                    ),
                                    rx.vstack(
                                        rx.box(
                                            rx.hstack(
                                                rx.text('', margin_left='1px'),
                                                rx.text("3", size="9", text_align="center"),
                                                rx.text('', margin_right='1px'),
                                            ),
                                            background_color="var(--plum-3)",
                                            border_radius="10px",
                                            padding_x="10px",
                                            padding_y="2px",
                                            text_align="center"
                                        ),
                                        align_items="center",
                                        spacing="0.3rem"
                                    ),

                                    spacing="2rem",
                                    align_items="center",
                                ),
                            ),
                            spacing="2",
                        ),
                        size="3",
                    ),                    
                    rx.card(
                        rx.flex(
                            rx.box(
                                rx.hstack(
                                    rx.text(
                                        "GPU 카드 개수",
                                        trim="bold",
                                        font_size="3rem",
                                        weight="both"
                                    ),
                                    rx.vstack(
                                        rx.box(
                                            rx.hstack(
                                                rx.text('', margin_left='1px'),
                                                rx.text(State.gpu_count, size="9", text_align="center"),
                                                rx.text('', margin_right='1px'),
                                            ),
                                            background_color="var(--plum-3)",
                                            border_radius="10px",
                                            padding_x="10px",
                                            padding_y="2px",
                                            text_align="center"
                                        ),
                                        align_items="center",
                                        spacing="0.3rem"
                                    ),

                                    spacing="2rem",
                                    align_items="center",
                                ),
                            ),
                            spacing="2",
                        ),
                        size="3",
                    ),                    
                    spacing='8'
                ),
                rx.box(height="5px"),
                rx.table.root(
                    rx.table.header(
                        rx.table.row(
                            rx.table.column_header_cell("GPU 노드 이름"),
                            rx.table.column_header_cell("GPU 노드 상태"),
                            rx.table.column_header_cell("GPU 노드 아이피"),
                            rx.table.column_header_cell("GPU 노드 운영체제"),
                            rx.table.column_header_cell("GPU 노드 카드 종류"),
                            rx.table.column_header_cell("GPU 노드 사용 여부"),
                        ),
                    ),
                    rx.table.body(
                        rx.table.row(
                            rx.table.row_header_cell("Kubernetes Cluster"),
                            rx.table.cell(rx.badge("활성화", color='white', background_color='green', size='2', variant="solid")),
                            rx.table.cell("10.40.217.244"),
                            rx.table.cell("Ubuntu 20.04.6 LTS"),
                            rx.table.cell("NVIDIA RTX 3090"),
                            rx.table.cell(
                                rx.badge(
                                    State.mem_status_0,
                                    color="white",
                                    background_color=rx.cond(State.mem_used_percents_0 > gpu_percent_0, "blue", "red"),
                                    size="2",
                                    variant="solid"
                                ),
                                text_align="left",
                            ),
                        ),
                        rx.table.row(
                            rx.table.row_header_cell("Kubernetes Client 1"),
                            rx.table.cell(rx.badge("활성화", color='white', background_color='green', size='2', variant="solid")),
                            rx.table.cell("10.40.217.236"),
                            rx.table.cell("Ubuntu 20.04.6 LTS"),
                            rx.table.cell("NVIDIA GeForce GTX 1080"),
                            rx.table.cell(
                                rx.badge(
                                    State.mem_status_1,
                                    color="white",
                                    background_color=rx.cond(State.mem_used_percents_1 > gpu_percent_1, "blue", "red"),
                                    size="2",
                                    variant="solid"
                                ),
                                text_align="left",
                            ),                            
                        ),
                        rx.table.row(
                            rx.table.row_header_cell("Kubernetes Client 2"),
                            rx.table.cell(rx.badge("비활성화", color='black', background_color='orange', size='2', variant="solid")),
                            rx.table.cell("10.40.217.192"),
                            rx.table.cell("Ubuntu 22.04.4 LTS"),
                            rx.table.cell("NVIDIA GeForce GTX 1080"),
                            rx.table.cell(''),
                        ),                        
                    ),
                    width="100%",
                    size='2'
                ),                
                rx.box(height="15px"),
                rx.badge(rx.text(rx.text.strong("NVIDIA RTX 3090")), size="3", variant='surface'),
                rx.hstack(
                    rx.text(rx.text.strong(f"GPU 카드 {State.mem_gpu_indexs_0}번 사용량"), white_space="nowrap"),
                    rx.progress(value=State.mem_useds_0-gpu_memory_0, max=State.mem_totals_0-gpu_memory_0, width="600px"),
                    rx.text(
                        rx.text.strong(
                            #f"{State.mem_useds_0}/{State.mem_totals_0}MiB({State.mem_used_percents_0}%)"
                            f"{State.mem_useds_0-gpu_memory_0}/{State.mem_totals_0-gpu_memory_0}MiB({State.mem_used_percents_0_fmt}%)"
                        ),
                        white_space="nowrap",
                    ),
                ),
                rx.box(height="15px"),
                rx.badge(rx.text(rx.text.strong("NVIDIA GeForce GTX 1080")), size="3", variant='surface'),
                rx.hstack(
                    rx.text(rx.text.strong(f"GPU 카드 {State.mem_gpu_indexs_1}번 사용량"), white_space="nowrap"),
                    rx.progress(value=State.mem_useds_1-gpu_memory_1, max=State.mem_totals_1-gpu_memory_1, width="600px"),
                    rx.text(
                        rx.text.strong(
                            #f"{State.mem_useds_1}/{State.mem_totals_1}MiB({State.mem_used_percents_1}%)"
                            f"{State.mem_useds_1-gpu_memory_1}/{State.mem_totals_1-gpu_memory_1}MiB({State.mem_used_percents_1_fmt}%)"
                        ),
                        white_space="nowrap",
                    ),
                ),
                rx.box(height="15px"),
                rx.badge(rx.text(rx.text.strong("NVIDIA GeForce GTX 1080")), size="3", variant='surface'),                
            ),
            spacing="1em",
            align_items="flex-start",
        ),
        padding="2em",
        width="100%",
        height="100vh",            
    )

app = rx.App()
app.add_page(index, title='GPU Monitoring')
