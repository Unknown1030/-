import streamlit as st
import numpy as np
from dataclasses import dataclass
from typing import Dict, List
from enum import Enum
import pandas as pd

# 页面配置
st.set_page_config(
    page_title="朋友圈分组决策助手",
    page_icon="📱",
    layout="wide"
)

# 自定义样式
st.markdown("""
<style>
    .main {background-color: #f5f5f5;}
    .stButton>button {background-color: #07c160; color: white; border-radius: 20px;}
    .high-score {color: #07c160; font-size: 24px; font-weight: bold;}
    .medium-score {color: #faad14; font-size: 24px; font-weight: bold;}
    .low-score {color: #ff4d4f; font-size: 24px; font-weight: bold;}
    .decision-box {padding: 20px; border-radius: 10px; margin: 10px 0;}
    .recommend {background-color: #f6ffed; border: 1px solid #b7eb8f;}
    .warning {background-color: #fffbe6; border: 1px solid #ffe58f;}
    .danger {background-color: #fff1f0; border: 1px solid #ffa39e;}
</style>
""", unsafe_allow_html=True)


# 枚举定义
class ContentType(Enum):
    ACHIEVEMENT = "🏆 成就展示（获奖/升职/买房等）"
    DAILY = "🍜 日常生活（美食/风景/日常）"
    OPINION = "💭 观点表达（评论/转发/观点）"
    EMOTION = "💔 情感抒发（感慨/脆弱/情绪）"
    ENTERTAINMENT = "🎮 娱乐休闲（游戏/综艺/梗）"


class EmotionTone(Enum):
    POSITIVE = "😊 积极（开心/庆祝/满足）"
    NEUTRAL = "😐 中性（记录/陈述/平淡）"
    NEGATIVE = "😢 消极（抱怨/焦虑/沮丧）"
    CONTROVERSIAL = "🔥 争议（敏感话题/对立观点）"


class PersonType(Enum):
    TEACHER = "👨‍🏫 老师/导师"
    BOSS = "👔 老板/上级"
    PEER_SENIOR = "👫 同辈哥哥姐姐"
    CLASSMATE = "🎓 同学/同侪"
    RELATIVE = "👨‍👩‍👧 亲戚"
    PARENT = "👨‍👩‍👦 父母"
    FRIEND_CLOSE = "❤️ 亲密朋友"
    ACQUAINTANCE = "👋 普通熟人"


@dataclass
class RelationshipProfile:
    person_type: PersonType
    emotional_depth: float
    resource_value: float
    maintenance_cost: float
    defense_sensitivity: float
    cognitive_openness: float
    current_life_satisfaction: float
    last_interaction_days: int


@dataclass
class ContentProfile:
    content_type: ContentType
    emotion_tone: EmotionTone
    information_novelty: float
    aesthetic_quality: float
    self_exposure_level: float
    comparison_trigger: float
    value_alignment_risk: float


# 默认人群画像
DEFAULT_PROFILES = {
    PersonType.TEACHER: RelationshipProfile(
        person_type=PersonType.TEACHER,
        emotional_depth=0.6, resource_value=0.8,
        maintenance_cost=0.4, defense_sensitivity=0.3,
        cognitive_openness=0.7, current_life_satisfaction=0.7,
        last_interaction_days=30
    ),
    PersonType.BOSS: RelationshipProfile(
        person_type=PersonType.BOSS,
        emotional_depth=0.3, resource_value=0.9,
        maintenance_cost=0.6, defense_sensitivity=0.4,
        cognitive_openness=0.5, current_life_satisfaction=0.6,
        last_interaction_days=7
    ),
    PersonType.PEER_SENIOR: RelationshipProfile(
        person_type=PersonType.PEER_SENIOR,
        emotional_depth=0.7, resource_value=0.6,
        maintenance_cost=0.5, defense_sensitivity=0.6,
        cognitive_openness=0.6, current_life_satisfaction=0.5,
        last_interaction_days=14
    ),
    PersonType.CLASSMATE: RelationshipProfile(
        person_type=PersonType.CLASSMATE,
        emotional_depth=0.6, resource_value=0.5,
        maintenance_cost=0.3, defense_sensitivity=0.7,
        cognitive_openness=0.6, current_life_satisfaction=0.5,
        last_interaction_days=60
    ),
    PersonType.RELATIVE: RelationshipProfile(
        person_type=PersonType.RELATIVE,
        emotional_depth=0.5, resource_value=0.4,
        maintenance_cost=0.7, defense_sensitivity=0.5,
        cognitive_openness=0.3, current_life_satisfaction=0.6,
        last_interaction_days=90
    ),
    PersonType.PARENT: RelationshipProfile(
        person_type=PersonType.PARENT,
        emotional_depth=0.9, resource_value=0.3,
        maintenance_cost=0.8, defense_sensitivity=0.8,
        cognitive_openness=0.2, current_life_satisfaction=0.5,
        last_interaction_days=3
    ),
    PersonType.FRIEND_CLOSE: RelationshipProfile(
        person_type=PersonType.FRIEND_CLOSE,
        emotional_depth=0.9, resource_value=0.7,
        maintenance_cost=0.4, defense_sensitivity=0.4,
        cognitive_openness=0.8, current_life_satisfaction=0.6,
        last_interaction_days=5
    ),
    PersonType.ACQUAINTANCE: RelationshipProfile(
        person_type=PersonType.ACQUAINTANCE,
        emotional_depth=0.2, resource_value=0.3,
        maintenance_cost=0.2, defense_sensitivity=0.5,
        cognitive_openness=0.5, current_life_satisfaction=0.5,
        last_interaction_days=180
    ),
}

# 冲突矩阵
CONTENT_CONFLICT_MATRIX = {
    (ContentType.ACHIEVEMENT, PersonType.BOSS): 0.3,
    (ContentType.ACHIEVEMENT, PersonType.CLASSMATE): 0.7,
    (ContentType.ACHIEVEMENT, PersonType.PEER_SENIOR): 0.5,
    (ContentType.ACHIEVEMENT, PersonType.PARENT): 0.2,
    (ContentType.EMOTION, PersonType.BOSS): 0.8,
    (ContentType.EMOTION, PersonType.TEACHER): 0.4,
    (ContentType.EMOTION, PersonType.PARENT): 0.6,
    (ContentType.OPINION, PersonType.BOSS): 0.6,
    (ContentType.OPINION, PersonType.RELATIVE): 0.7,
    (ContentType.ENTERTAINMENT, PersonType.TEACHER): 0.4,
    (ContentType.ENTERTAINMENT, PersonType.BOSS): 0.5,
}


class WeChatDecisionEngine:
    def __init__(self):
        self.profiles = DEFAULT_PROFILES.copy()

    def calculate_suitability(self, content: ContentProfile, person: RelationshipProfile) -> Dict:
        # 1. 关系价值系数 RV
        rv = (person.emotional_depth * 0.4 +
              person.resource_value * 0.35 +
              (1 - person.maintenance_cost) * 0.25)

        # 2. 内容价值密度 CV
        emotional_resonance = self._estimate_emotional_resonance(content, person)
        cv = (content.information_novelty * 0.4 +
              emotional_resonance * 0.35 +
              content.aesthetic_quality * 0.25)

        # 3. 情绪安全度 ES
        threat_perception = self._calculate_threat(content, person)
        self_enhancement = self._calculate_enhancement(content, person)
        defense_coef = person.defense_sensitivity
        es = max(0, 1 - abs(threat_perception - self_enhancement) * defense_coef)

        # 4. 认知摩擦成本 CF
        unfamiliarity = 1 - person.cognitive_openness if content.content_type == ContentType.OPINION else 0.3
        value_conflict = content.value_alignment_risk
        cf = unfamiliarity * 0.5 + value_conflict * 0.5

        # 5. 社会风险暴露 SR
        privacy_risk = content.self_exposure_level
        image_risk = self._calculate_image_risk(content, person)
        relationship_risk = threat_perception * person.defense_sensitivity
        sr = privacy_risk * 0.4 + image_risk * 0.35 + relationship_risk * 0.25

        # 6. 情境调节因子 SF
        sf = self._situation_factor(content, person)

        # 7. 基础适配性计算
        numerator = rv * cv * es
        denominator = max(cf * sr, 0.01)

        base_suitability = (numerator / denominator) * sf
        suitability = max(0, min(1, base_suitability))

        # 8. 决策建议
        decision = self._generate_decision(suitability, content, person)

        return {
            "suitability_percentage": round(suitability * 100, 1),
            "decision": decision["action"],
            "decision_class": decision["decision_class"],
            "confidence": decision["confidence"],
            "breakdown": {
                "关系价值系数": round(rv, 3),
                "内容价值密度": round(cv, 3),
                "情绪安全度": round(es, 3),
                "认知摩擦成本": round(cf, 3),
                "社会风险暴露": round(sr, 3),
                "情境调节因子": round(sf, 3)
            },
            "risk_flags": decision["risks"],
            "optimization_tips": decision["tips"]
        }

    def _estimate_emotional_resonance(self, content: ContentProfile, person: RelationshipProfile) -> float:
        base = 0.5

        if content.emotion_tone == EmotionTone.POSITIVE and person.current_life_satisfaction > 0.6:
            base += 0.2
        elif content.emotion_tone == EmotionTone.NEGATIVE and person.current_life_satisfaction < 0.4:
            base += 0.15
        elif content.emotion_tone == EmotionTone.CONTROVERSIAL:
            base -= 0.3

        if content.content_type == ContentType.DAILY and person.cognitive_openness < 0.4:
            base += 0.1
        if content.content_type == ContentType.OPINION and person.cognitive_openness > 0.7:
            base += 0.15

        return max(0, min(1, base))

    def _calculate_threat(self, content: ContentProfile, person: RelationshipProfile) -> float:
        base_threat = content.comparison_trigger
        recency_factor = max(0, 1 - person.last_interaction_days / 365)
        similarity_boost = 0.2 if person.person_type in [PersonType.CLASSMATE, PersonType.PEER_SENIOR] else 0

        threat = (base_threat + similarity_boost) * (0.5 + person.defense_sensitivity * 0.5)
        threat *= (1 + recency_factor * 0.3)

        return min(1, threat)

    def _calculate_enhancement(self, content: ContentProfile, person: RelationshipProfile) -> float:
        if content.content_type == ContentType.ACHIEVEMENT:
            if person.person_type in [PersonType.PARENT, PersonType.TEACHER, PersonType.FRIEND_CLOSE]:
                return 0.6
            else:
                return 0.2
        elif content.content_type == ContentType.EMOTION and content.emotion_tone == EmotionTone.NEGATIVE:
            return 0.4 if person.defense_sensitivity > 0.5 else 0.2
        elif content.content_type == ContentType.DAILY:
            return 0.3

        return 0.2

    def _calculate_image_risk(self, content: ContentProfile, person: RelationshipProfile) -> float:
        conflict_key = (content.content_type, person.person_type)
        base_risk = CONTENT_CONFLICT_MATRIX.get(conflict_key, 0.3)

        if content.emotion_tone == EmotionTone.NEGATIVE and person.person_type in [PersonType.BOSS, PersonType.TEACHER]:
            base_risk += 0.3

        return min(1, base_risk)

    def _situation_factor(self, content: ContentProfile, person: RelationshipProfile) -> float:
        sf = 1.0

        if person.last_interaction_days > 90:
            sf *= 0.8

        if person.person_type == PersonType.PARENT and person.last_interaction_days < 7:
            sf *= 1.2

        return max(0.5, min(1.5, sf))

    def _generate_decision(self, suitability: float, content: ContentProfile, person: RelationshipProfile) -> Dict:
        if suitability >= 0.8:
            action = "✅ 强烈推荐可见"
            decision_class = "recommend"
            confidence = "高"
        elif suitability >= 0.6:
            action = "🟢 建议可见，可优化文案"
            decision_class = "recommend"
            confidence = "中高"
        elif suitability >= 0.4:
            action = "⚪ 边缘地带，建议延迟或修改"
            decision_class = "warning"
            confidence = "中"
        elif suitability >= 0.2:
            action = "🟡 谨慎屏蔽，除非特殊目的"
            decision_class = "warning"
            confidence = "中低"
        else:
            action = "🔴 坚决屏蔽"
            decision_class = "danger"
            confidence = "高"

        risks = []
        if content.comparison_trigger > 0.7:
            risks.append("高嫉妒风险")
        if content.self_exposure_level > 0.7:
            risks.append("隐私泄露风险")
        if content.value_alignment_risk > 0.6:
            risks.append("价值观冲突")
        if person.defense_sensitivity > 0.7 and content.content_type == ContentType.ACHIEVEMENT:
            risks.append("对方高敏感+成就展示=危险组合")

        tips = []
        if suitability < 0.6 and content.content_type == ContentType.ACHIEVEMENT:
            tips.append("增加过程描述，减少结果炫耀")
            tips.append("强调'幸运'、'贵人相助'等去个人化因素")
        if suitability < 0.6 and content.emotion_tone == EmotionTone.NEGATIVE:
            tips.append("转为私聊倾诉，而非公开朋友圈")
            tips.append("如必须发，增加'已解决'的结尾，减少担忧")
        if person.person_type == PersonType.BOSS and content.content_type == ContentType.ENTERTAINMENT:
            tips.append("改为展示'充电学习'而非'纯娱乐'")
        if suitability < 0.4:
            tips.append("建议延迟发布，等待更好时机")

        return {
            "action": action,
            "decision_class": decision_class,
            "confidence": confidence,
            "risks": risks,
            "tips": tips
        }


# 初始化引擎
engine = WeChatDecisionEngine()

# ==================== UI 界面 ====================

st.title("📱 朋友圈分组决策助手")
st.markdown("基于社交心理学算法，智能分析你的朋友圈内容适合展示给哪些人")

# 侧边栏 - 内容输入
with st.sidebar:
    st.header("📝 内容设置")

    content_type = st.selectbox(
        "选择内容类型",
        list(ContentType),
        format_func=lambda x: x.value
    )

    emotion_tone = st.selectbox(
        "选择情绪基调",
        list(EmotionTone),
        format_func=lambda x: x.value
    )

    st.subheader("内容属性评分 (0-100)")

    info_novelty = st.slider("信息新颖度", 0, 100, 60,
                             help="这条内容对看的人来说有多新鲜？")
    aesthetic = st.slider("审美质量", 0, 100, 50,
                          help="图片/文字的美感程度")
    self_exposure = st.slider("自我暴露程度", 0, 100, 40,
                              help="暴露了多少隐私/脆弱？")
    comparison = st.slider("社会比较触发度", 0, 100, 30,
                           help="容易引发'他比我好/差'的比较吗？")
    value_risk = st.slider("价值观冲突风险", 0, 100, 20,
                           help="与对方价值观冲突的可能性")

    # 快捷预设
    st.subheader("⚡ 快捷预设")
    preset = st.selectbox("选择场景预设",
                          ["无", "刚失恋求安慰", "升职加薪", "深夜emo",
                           "旅游打卡", "吐槽工作", "晒娃", "政治观点"])

    if preset == "刚失恋求安慰":
        content_type = ContentType.EMOTION
        emotion_tone = EmotionTone.NEGATIVE
        info_novelty = 80
        aesthetic = 30
        self_exposure = 90
        comparison = 10
        value_risk = 30
        st.success("已加载预设：情感抒发-消极-高暴露")
    elif preset == "升职加薪":
        content_type = ContentType.ACHIEVEMENT
        emotion_tone = EmotionTone.POSITIVE
        info_novelty = 70
        aesthetic = 40
        self_exposure = 50
        comparison = 90
        value_risk = 20

# 主界面 - 分析结果
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎯 目标人群分析")

    # 选择要分析的人群
    selected_groups = st.multiselect(
        "选择要分析的人群（可多选）",
        list(PersonType),
        default=[PersonType.PARENT, PersonType.BOSS, PersonType.FRIEND_CLOSE, PersonType.CLASSMATE],
        format_func=lambda x: x.value
    )

    # 自定义人群状态
    with st.expander("🔧 高级设置：自定义对方状态"):
        custom_satisfaction = st.slider("对方当前生活满意度", 0.0, 1.0, 0.5, 0.1,
                                        help="对方最近过得怎么样？")
        custom_defense = st.slider("对方防御敏感度", 0.0, 1.0, 0.5, 0.1,
                                   help="对方容易嫉妒/焦虑吗？")
        custom_recent = st.slider("上次互动（天前）", 0, 365, 30,
                                  help="多久没联系了？")

with col2:
    st.subheader("📊 一键分析结果")

    if st.button("🚀 开始分析", use_container_width=True):
        if not selected_groups:
            st.warning("请至少选择一个人群进行分析")
        else:
            # 创建内容档案
            content = ContentProfile(
                content_type=content_type,
                emotion_tone=emotion_tone,
                information_novelty=info_novelty / 100,
                aesthetic_quality=aesthetic / 100,
                self_exposure_level=self_exposure / 100,
                comparison_trigger=comparison / 100,
                value_alignment_risk=value_risk / 100
            )

            results = []

            for group in selected_groups:
                # 获取基础画像并应用自定义
                person = DEFAULT_PROFILES[group]
                person = RelationshipProfile(
                    person_type=group,
                    emotional_depth=person.emotional_depth,
                    resource_value=person.resource_value,
                    maintenance_cost=person.maintenance_cost,
                    defense_sensitivity=custom_defense if 'custom_defense' in locals() else person.defense_sensitivity,
                    cognitive_openness=person.cognitive_openness,
                    current_life_satisfaction=custom_satisfaction if 'custom_satisfaction' in locals() else person.current_life_satisfaction,
                    last_interaction_days=custom_recent if 'custom_recent' in locals() else person.last_interaction_days
                )

                result = engine.calculate_suitability(content, person)
                results.append({
                    "人群": group.value,
                    "适配性": result["suitability_percentage"],
                    "决策": result["decision"],
                    "置信度": result["confidence"],
                    "风险": ", ".join(result["risk_flags"]) if result["risk_flags"] else "无",
                    "建议": result["optimization_tips"][0] if result["optimization_tips"] else "无需优化",
                    "详细数据": result
                })

            # 排序：适配性从高到低
            results.sort(key=lambda x: x["适配性"], reverse=True)

            # 显示结果
            for i, res in enumerate(results):
                score = res["适配性"]
                decision_class = res["详细数据"]["decision_class"]

                # 颜色编码
                if score >= 60:
                    score_class = "high-score"
                    border_color = "#b7eb8f"
                elif score >= 40:
                    score_class = "medium-score"
                    border_color = "#ffe58f"
                else:
                    score_class = "low-score"
                    border_color = "#ffa39e"

                with st.container():
                    st.markdown(f"""
                    <div style="border: 2px solid {border_color}; border-radius: 10px; padding: 15px; margin: 10px 0; background-color: white;">
                        <div style="display: flex; justify-content: space-between; align-items: center;">
                            <h4 style="margin: 0;">{res["人群"]}</h4>
                            <span class="{score_class}">{score}%</span>
                        </div>
                        <p style="margin: 5px 0; color: #666; font-size: 14px;">
                            {res["决策"]} | 置信度: {res["置信度"]}
                        </p>
                        <p style="margin: 5px 0; color: #ff4d4f; font-size: 13px;">
                            ⚠️ 风险: {res["风险"]}
                        </p>
                        <p style="margin: 5px 0; color: #1890ff; font-size: 13px;">
                            💡 {res["建议"]}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                    # 展开查看详细指标
                    with st.expander("查看详细指标"):
                        breakdown = res["详细数据"]["breakdown"]
                        cols = st.columns(3)
                        cols[0].metric("关系价值", f"{breakdown['关系价值系数']:.2f}")
                        cols[1].metric("内容价值", f"{breakdown['内容价值密度']:.2f}")
                        cols[2].metric("情绪安全", f"{breakdown['情绪安全度']:.2f}")

                        cols = st.columns(3)
                        cols[0].metric("认知摩擦", f"{breakdown['认知摩擦成本']:.2f}")
                        cols[1].metric("社会风险", f"{breakdown['社会风险暴露']:.2f}")
                        cols[2].metric("情境因子", f"{breakdown['情境调节因子']:.2f}")

# 底部 - 智能建议汇总
st.markdown("---")
st.subheader("🎓 智能策略建议")

if 'results' in locals() and results:
    visible_groups = [r for r in results if r["适配性"] >= 60]
    hidden_groups = [r for r in results if r["适配性"] < 40]
    edge_groups = [r for r in results if 40 <= r["适配性"] < 60]

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### ✅ 建议可见")
        if visible_groups:
            for g in visible_groups:
                st.success(f"**{g['人群']}** - {g['适配性']}%")
        else:
            st.info("无")

    with col2:
        st.markdown("#### ⚠️ 边缘地带")
        if edge_groups:
            for g in edge_groups:
                st.warning(f"**{g['人群']}** - {g['适配性']}%")
        else:
            st.info("无")

    with col3:
        st.markdown("#### 🚫 建议屏蔽")
        if hidden_groups:
            for g in hidden_groups:
                st.error(f"**{g['人群']}** - {g['适配性']}%")
        else:
            st.info("无")

    # 文案优化建议
    st.markdown("#### ✍️ 文案优化模板")

    if content_type == ContentType.ACHIEVEMENT and any(r["适配性"] < 60 for r in results):
        st.info("""
        **原问题**: 成就展示易引发比较和嫉妒

        **优化模板**:
        > "收到新offer，感恩这一路的贵人相助。从投简历到面试，经历过自我怀疑，也收获了很多意外支持。
        > 特别感谢[具体人名]在[具体环节]的帮助。新起点，继续踏实做事。"

        **关键技巧**:
        - 强调过程而非结果
        - 点名感谢去个人化
        - 提及困难建立共情
        """)

    elif content_type == ContentType.EMOTION and emotion_tone == EmotionTone.NEGATIVE:
        st.info("""
        **原问题**: 消极情绪易引发担忧或被视为不成熟

        **优化模板**:
        > "最近在经历一些个人生活的调整，情绪像过山车。但已经开始看到隧道尽头的光，
        > 也在重新理解什么是真正重要的。感谢默默关心的朋友们，给我一点时间。"

        **关键技巧**:
        - 承认情绪但暗示好转
        - 不透露具体细节
        - 感谢已收到的支持（预设有人关心）
        """)

    elif content_type == ContentType.OPINION:
        st.info("""
        **原问题**: 观点表达易引发价值观冲突

        **优化模板**:
        > "最近观察到[现象]，产生了一些想法。可能不完全正确，但想记录下来：
        > [温和的观点陈述]。很想听听不同视角的朋友怎么看？"

        **关键技巧**:
        - 软化表达（"可能不完全正确"）
        - 邀请对话而非宣告结论
        - 避免绝对化词汇
        """)

else:
    st.info("点击左侧'开始分析'按钮获取智能建议")

# 使用说明
with st.expander("📖 使用说明 & 算法说明"):
    st.markdown("""
    ### 核心公式
    ```
    适配性 = (关系价值 × 内容价值 × 情绪安全) / (认知摩擦 × 社会风险) × 情境调节
    ```

    ### 评分标准
    - **80-100%**: 强烈推荐，内容与人高度匹配
    - **60-79%**: 建议可见，可小幅优化文案
    - **40-59%**: 边缘地带，建议延迟或修改内容
    - **20-39%**: 谨慎屏蔽，除非有特殊目的
    - **0-19%**: 坚决屏蔽，风险远大于收益

    ### 人群画像说明
    系统预设了8类典型人群的默认参数，你可以在"高级设置"中调整：
    - **防御敏感度**: 对方容易嫉妒/焦虑的程度
    - **生活满意度**: 对方当前的生活状态
    - **上次互动**: 心理距离的重要指标

    ### 隐私声明
    所有分析在本地完成，不会上传任何数据。
    """)