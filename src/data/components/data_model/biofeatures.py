from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class TmSequences:
    """Represents a collection of Tm sequences used in the preprocessing of pe6 data.

    Attributes:
        Tm1_PBS (str): The Tm sequence for PBS.
        Tm2_RTT_cTarget_sameLength (str): The Tm sequence for RTT with the same length as cTarget.
        Tm3_RTT_cTarget_replaced (str): The Tm sequence for RTT with cTarget replaced.
        Tm4_cDNA_PAM_oppositeTarget (Optional[Tuple[str, str]]): The Tm sequence for cDNA with PAM opposite the target.
        Tm5_RTT_cDNA (str): The Tm sequence for RTT with cDNA.
    """

    Tm1_PBS_seq: str
    Tm2_RTT_cTarget_sameLength_seq: str
    Tm3_RTT_cTarget_replaced_seq: str
    Tm4_cDNA_PAM_oppositeTarget_seq: Optional[Tuple[str, str]]
    Tm5_RTT_cDNA_seq: str


@dataclass
class TmData:
    """Represents a collection of melting temperature data for various targets.

    Attributes:
        Tm1_PBS (float): The melting temperature for the PBS target.
        Tm2_RTT_cTarget_sameLength (float): The melting temperature for the RTT_cTarget_sameLength target.
        Tm3_RTT_cTarget_replaced (float): The melting temperature for the RTT_cTarget_replaced target.
        Tm4_cDNA_PAM_oppositeTarget (float): The melting temperature for the cDNA_PAM_oppositeTarget target.
        Tm5_RTT_cDNA (float): The melting temperature for the RTT_cDNA target.
        deltaTm_Tm4_Tm2 (float): The difference in melting temperatures between Tm4_cDNA_PAM_oppositeTarget and Tm2_RTT_cTarget_sameLength.
    """

    Tm1_PBS: float
    Tm2_RTT_cTarget_sameLength: float
    Tm3_RTT_cTarget_replaced: float
    Tm4_cDNA_PAM_oppositeTarget: float
    Tm5_RTT_cDNA: float
    deltaTm_Tm4_Tm2: float


@dataclass
class PegRNAExtensionData:
    """Represents the extension data for a pegRNA.

    Attributes:
        GC_count_PBS (int): The GC count of the PBS region.
        GC_count_RTT (int): The GC count of the RTT region.
        GC_count_RT_PBS (int): The GC count of the RT-PBS region.
        GC_contents_PBS (float): The GC content of the PBS region.
        GC_contents_RTT (float): The GC content of the RTT region.
        GC_contents_RT_PBS (float): The GC content of the RT-PBS region.
    """

    GC_count_PBS: int
    GC_count_RTT: int
    GC_count_RT_PBS: int
    GC_contents_PBS: float
    GC_contents_RTT: float
    GC_contents_RT_PBS: float


@dataclass
class MFEData:
    """Represents the minimum free energy data for a pegRNA.

    Attributes:
        MFE_PBS (float): The minimum free energy of the PBS region.
        MFE_RTT (float): The minimum free energy of the RTT region.
        MFE_RT_PBS (float): The minimum free energy of the RT-PBS region.
    """

    MFE_RT_PBS_polyT: float
    MFE_Spacer: float


@dataclass
class EditTypeClass(Tuple[int, int, int]):
    """Represents the types of edits that can be performed on a biofeature.

    Attributes:
        type_sub (int): Indicates whether substitution edit is allowed.
        type_ins (int): Indicates whether insertion edit is allowed.
        type_del (int): Indicates whether deletion edit is allowed.
    """

    type_sub: int
    type_ins: int
    type_del: int


@dataclass
class TargetSequenceData:
    wild_type_sequence: str
    deepspcas9_guide_30: str
    prime_edited_sequence: str
    edit_position: int
