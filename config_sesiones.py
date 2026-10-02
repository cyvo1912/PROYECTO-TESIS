SESIONES = [
    {
        "sujeto": "GRABACION 1",
        "json": r"D:\CODIGO SEMINARIO II\dmd-dataset-distraction-gB-6\dmd\gB\6\s2\gB_6_s2_2019-03-11T13;46;14+01;00_rgb_ann_distraction.json",
    },
    {
        "sujeto": "GRABACION 2",
        "json": r"D:\CODIGO SEMINARIO II\dmd-dataset-distraction-gB-7\dmd\gB\7\s2\gB_7_s2_2019-03-11T14;12;25+01;00_rgb_ann_distraction.json",
    },
    {
        "sujeto": "GRABACION 3",
        "json": r"D:\CODIGO SEMINARIO II\dmd-dataset-distraction-gB-9\dmd\gB\9\s2\gB_9_s2_2019-03-07T16;21;20+01;00_rgb_ann_distraction.json",
    },
    {
        "sujeto": "GRABACION 4",
        "json": r"D:\CODIGO SEMINARIO II\dmd-dataset-distraction-gE-28\dmd\gE\28\s2\gE_28_s2_2019-03-15T10;12;30+01;00_rgb_ann_distraction.json",
    },
    {
        "sujeto": "GRABACION 5",
        "json": r"D:\CODIGO SEMINARIO II\dmd-dataset-distraction-gE-29\dmd\gE\29\s2\gE_29_s2_2019-03-15T13;42;24+01;00_rgb_ann_distraction.json",
    },
    {
        "sujeto": "GRABACION 6",
        "json": r"D:\CODIGO SEMINARIO II\dmd-dataset-distraction-gA-1\dmd\gA\1\s2\gA_1_s2_2019-03-08T09;21;03+01;00_rgb_ann_distraction.json",
    },
    {
        "sujeto": "GRABACION 7",
        "json": r"D:\CODIGO SEMINARIO II\dmd-dataset-distraction-gA-5\dmd\gA\5\s2\gA_5_s2_2019-03-08T10;46;46+01;00_rgb_ann_distraction.json",
    },
    {
        "sujeto": "GRABACION 8",
        "json": r"D:\CODIGO SEMINARIO II\dmd-dataset-distraction-gC-13\dmd\gC\13\s2\gC_13_s2_2019-03-04T10;11;37+01;00_rgb_ann_distraction.json",
    },
    {
        "sujeto": "GRABACION 9",
        "json": r"D:\CODIGO SEMINARIO II\dmd-dataset-distraction-gZ-37\dmd\gZ\37\s2\gZ_37_s2_2019-04-08T15;45;15+02;00_rgb_ann_distraction.json",
    },
    {
        "sujeto": "GRABACION 10",
        "json": r"D:\CODIGO SEMINARIO II\dmd-dataset-distraction-gZ-36\dmd\gZ\36\s2\gZ_36_s2_2019-04-09T10;39;38+02;00_rgb_ann_distraction.json",
    },
    {
        "sujeto": "GRABACION 11",
        "json": r"D:\CODIGO SEMINARIO II\dmd-dataset-distraction-gZ-33\dmd\gZ\33\s2\gZ_33_s2_2019-04-08T09;59;25+02;00_rgb_ann_distraction.json",
    },
    {
        "sujeto": "GRABACION 12",
        "json": r"D:\CODIGO SEMINARIO II\dmd-dataset-distraction-gF-23\dmd\gF\23\s2\gF_23_s2_2019-03-04T16;09;26+01;00_rgb_ann_distraction.json",
    },
    {
        "sujeto": "GRABACION 13",
        "json": r"D:\CODIGO SEMINARIO II\dmd-dataset-distraction-gC-14\dmd\gC\14\s2\gC_14_s2_2019-03-04T11;48;02+01;00_rgb_ann_distraction.json",
    },
    {
        "sujeto": "GRABACION 14",
        "json": r"D:\CODIGO SEMINARIO II\dmd-dataset-rgb_ir-gB-10\dmd\gB\10\s2\gB_10_s2_2019-03-11T15;15;21+01;00_rgb_ann_distraction.json",
    },
]


def sesiones_listas():
    listas = [s for s in SESIONES if s["json"] is not None]
    faltantes = [s["sujeto"] for s in SESIONES if s["json"] is None]

    if faltantes:
        print(f"[config_sesiones] Aún sin ruta, se omiten por ahora: {', '.join(faltantes)}")

    return listas